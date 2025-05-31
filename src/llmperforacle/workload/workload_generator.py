"""Workload generator for creating realistic LLM request streams."""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import simpy

from .models import ClientProfile, Request
from .sampler import DistributionSampler

logger = logging.getLogger(__name__)


class WorkloadGenerator:
    """Generates and dispatches LLM requests based on configured patterns."""
    
    def __init__(
        self,
        simpy_env: simpy.Environment,
        config: Dict[str, Any],
        target_frameworks_map: Dict[str, Union[simpy.Store, Any]],
        metrics_collector: Any,  # Will be MetricsCollector
        hardware_platform: Any,  # Will be VirtualHardwarePlatform
    ):
        """Initialize the workload generator.
        
        Args:
            simpy_env: SimPy environment instance
            config: Workload configuration containing:
                - total_duration: Simulation duration in seconds
                - client_profiles: List of client profile configurations
                - bytes_per_token_estimate_for_network: Network size per token
                - random_seed: Random seed for reproducibility
            target_frameworks_map: Map of framework_id to request queue/handler
            metrics_collector: Metrics collection instance
            hardware_platform: Virtual hardware platform instance
        """
        self.simpy_env = simpy_env
        self.config = config
        self.target_frameworks_map = target_frameworks_map
        self.metrics_collector = metrics_collector
        self.hardware_platform = hardware_platform
        
        # Initialize sampler with seed if provided
        seed = config.get("random_seed")
        self.sampler = DistributionSampler(seed)
        if seed is not None:
            random.seed(seed)
        
        # Parse client profiles
        self.client_profiles = self._parse_client_profiles(config.get("client_profiles", []))
        
        # Request tracking
        self.request_counter = 0
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Network configuration
        self.bytes_per_token = config.get("bytes_per_token_estimate_for_network", 2)
        
        # Data parallelism load balancing
        self.load_balancing_strategy = config.get("load_balancing_strategy", "round_robin")
        self.framework_request_counts = {fw_id: 0 for fw_id in target_frameworks_map.keys()}
        self._session_fw_mapping = {}  # Initialize session to framework mapping
        
        logger.info(
            f"WorkloadGenerator initialized with {len(self.client_profiles)} client profiles, "
            f"load balancing: {self.load_balancing_strategy}"
        )
    
    def _parse_client_profiles(self, profiles_config: List[Dict[str, Any]]) -> List[ClientProfile]:
        """Parse client profile configurations into ClientProfile objects."""
        profiles = []
        
        for profile_config in profiles_config:
            profile = ClientProfile(
                profile_name=profile_config["profile_name"],
                weight=profile_config.get("weight", 1.0),
                inter_arrival_time_dist_config=profile_config["inter_arrival_time_dist_config"],
                prompt_tokens_dist_config=profile_config["prompt_tokens_dist_config"],
                max_output_tokens_dist_config=profile_config["max_output_tokens_dist_config"],
                conversational_probability=profile_config.get("conversational_probability", 0.0),
                streaming_response_probability=profile_config.get("streaming_response_probability", 1.0),
                user_priority_dist_config=profile_config.get("user_priority_dist_config"),
                follow_up_inter_arrival_time_dist_config=profile_config.get(
                    "follow_up_inter_arrival_time_dist_config"
                ),
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_requests_process(self) -> simpy.events.Event:
        """Main SimPy process that generates requests according to workload patterns."""
        total_duration = self.config.get("total_duration", float("inf"))
        total_requests = self.config.get("total_requests_to_generate")
        
        logger.info(f"Starting workload generation (duration: {total_duration}s)")
        
        while True:
            # Check termination conditions
            if self.simpy_env.now >= total_duration:
                break
            if total_requests is not None and self.request_counter >= total_requests:
                break
            
            # Select client profile based on weights
            profile = self._select_client_profile()
            
            # Determine next request timing
            next_request_time, session_info = self._get_next_request_timing(profile)
            
            # Wait for the next request
            if next_request_time > 0:
                yield self.simpy_env.timeout(next_request_time)
            
            # Create and dispatch request
            request = self._create_request(profile, session_info)
            yield self.simpy_env.process(self._dispatch_request(request))
            
            # Update session state if conversational
            self._update_session_state(request, profile)
        
        logger.info(f"Workload generation completed. Generated {self.request_counter} requests")
    
    def _select_client_profile(self) -> ClientProfile:
        """Select a client profile based on configured weights."""
        if len(self.client_profiles) == 1:
            return self.client_profiles[0]
        
        weights = [p.weight for p in self.client_profiles]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return random.choices(self.client_profiles, weights=normalized_weights)[0]
    
    def _get_next_request_timing(
        self, profile: ClientProfile
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Determine timing for the next request, considering conversational follow-ups."""
        # Check for pending conversational follow-ups
        current_time = self.simpy_env.now
        
        for session_id, session_data in list(self.active_sessions.items()):
            if session_data.get("next_turn_time", float("inf")) <= current_time:
                # This session is ready for a follow-up
                return 0, {"session_id": session_id, "is_continuation": True}
        
        # No pending follow-ups, generate new request timing
        iat = self.sampler.sample(profile.inter_arrival_time_dist_config)
        return iat, {"session_id": None, "is_continuation": False}
    
    def _create_request(
        self, profile: ClientProfile, session_info: Dict[str, Any]
    ) -> Request:
        """Create a new request based on profile and session information."""
        self.request_counter += 1
        request_id = f"req_{self.request_counter}"
        
        # Determine session parameters
        if session_info["is_continuation"]:
            session_id = session_info["session_id"]
            is_conversational_turn = True
            session_data = self.active_sessions[session_id]
            client_id = session_data["client_id"]
            
            # For conversational turns, accumulate tokens to simulate growing context
            # The prompt includes: previous prompt + previous response + new user input
            new_user_input = self.sampler.sample(profile.prompt_tokens_dist_config)
            
            # Check if we should use accumulated tokens (for prefix caching simulation)
            if self.config.get("accumulate_conversational_tokens", True):
                # Accumulated tokens = previous prompt + previous response + new input
                prompt_num_tokens = (
                    session_data["accumulated_prompt_tokens"] + 
                    session_data["accumulated_output_tokens"] + 
                    new_user_input
                )
            else:
                # Use sampled value directly (old behavior)
                prompt_num_tokens = new_user_input
        else:
            session_id = f"sess_{self.request_counter}"
            is_conversational_turn = False
            client_id = f"client_{profile.profile_name}_{self.request_counter % 100}"
            # Sample request characteristics normally for new conversations
            prompt_num_tokens = self.sampler.sample(profile.prompt_tokens_dist_config)
        
        max_output_tokens = self.sampler.sample(profile.max_output_tokens_dist_config)
        streaming_response = random.random() < profile.streaming_response_probability
        
        # Sample priority if configured
        user_priority = 0
        if profile.user_priority_dist_config:
            user_priority = self.sampler.sample(profile.user_priority_dist_config)
        
        request = Request(
            request_id=request_id,
            client_id=client_id,
            session_id=session_id,
            arrival_time=self.simpy_env.now,
            prompt_num_tokens=prompt_num_tokens,
            max_output_tokens=max_output_tokens,
            is_conversational_turn=is_conversational_turn,
            streaming_response=streaming_response,
            user_priority=user_priority,
        )
        
        logger.debug(
            f"Created request {request_id}: {prompt_num_tokens} prompt tokens, "
            f"max {max_output_tokens} output tokens, conversational={is_conversational_turn}"
        )
        
        return request
    
    def _dispatch_request(self, request: Request) -> simpy.events.Event:
        """Dispatch a request to the target framework, including network simulation."""
        # Simulate network transfer of prompt
        prompt_data_size = request.prompt_num_tokens * self.bytes_per_token
        
        # Simulate client-to-server network transfer
        yield self.hardware_platform.submit_network_transfer_task(
            "client_node_0", "framework_entry_0", prompt_data_size
        )
        
        # Select target framework based on load balancing strategy
        target_fw_id = self._select_target_framework(request)
        if not target_fw_id:
            logger.error("No target frameworks available")
            return
        
        target = self.target_frameworks_map[target_fw_id]
        
        # Log request arrival
        self.metrics_collector.log_request_arrival(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            client_id=request.client_id,
            session_id=request.session_id,
            prompt_tokens=request.prompt_num_tokens,
            max_output=request.max_output_tokens,
        )
        
        # Dispatch to framework
        if isinstance(target, simpy.Store):
            # Queue-based dispatch
            yield target.put(request)
            self.metrics_collector.log_request_dispatch(
                request.request_id, self.simpy_env.now
            )
        else:
            # Direct handler dispatch
            if hasattr(target, "handle_incoming_request"):
                yield self.simpy_env.process(target.handle_incoming_request(request))
        
        logger.debug(f"Dispatched request {request.request_id} to {target_fw_id}")
    
    def _update_session_state(self, request: Request, profile: ClientProfile) -> None:
        """Update session state for conversational requests."""
        if not request.is_conversational_turn and random.random() < profile.conversational_probability:
            # Start a new conversation
            follow_up_iat = 5.0  # Default follow-up time
            if profile.follow_up_inter_arrival_time_dist_config:
                follow_up_iat = self.sampler.sample(
                    profile.follow_up_inter_arrival_time_dist_config
                )
            
            self.active_sessions[request.session_id] = {
                "client_id": request.client_id,
                "start_time": self.simpy_env.now,
                "next_turn_time": self.simpy_env.now + follow_up_iat,
                "turn_count": 1,
                "accumulated_prompt_tokens": request.prompt_num_tokens,
                "accumulated_output_tokens": request.max_output_tokens,  # Estimate
                "profile_name": profile.profile_name,
            }
        elif request.is_conversational_turn:
            # Update existing conversation
            session_data = self.active_sessions.get(request.session_id)
            if session_data:
                session_data["turn_count"] += 1
                
                # Update accumulated tokens for the next turn
                # The next prompt will include: current prompt + current output + new input
                # Since this is after the request is created, we update for the next turn
                session_data["accumulated_prompt_tokens"] = request.prompt_num_tokens
                session_data["accumulated_output_tokens"] = request.max_output_tokens  # Still an estimate
                
                # Decide if conversation continues
                max_turns = self.config.get("max_turns_per_session", 10)
                if session_data["turn_count"] >= max_turns or random.random() > 0.7:
                    # End conversation
                    del self.active_sessions[request.session_id]
                else:
                    # Schedule next turn
                    follow_up_iat = 5.0
                    if profile.follow_up_inter_arrival_time_dist_config:
                        follow_up_iat = self.sampler.sample(
                            profile.follow_up_inter_arrival_time_dist_config
                        )
                    session_data["next_turn_time"] = self.simpy_env.now + follow_up_iat
    
    def _select_target_framework(self, request: Request) -> Optional[str]:
        """Select target framework based on load balancing strategy.
        
        Supports:
        - round_robin: Distribute requests evenly in order
        - random: Random selection
        - least_loaded: Select framework with fewest pending requests (queue depth)
        - weighted_random: Random selection with configurable weights
        - session_affinity: Route conversational turns to same framework
        """
        framework_ids = list(self.target_frameworks_map.keys())
        if not framework_ids:
            return None
        
        if self.load_balancing_strategy == "round_robin":
            # Classic round-robin
            target_fw_id = framework_ids[self.request_counter % len(framework_ids)]
            
        elif self.load_balancing_strategy == "random":
            # Random selection
            target_fw_id = random.choice(framework_ids)
            
        elif self.load_balancing_strategy == "least_loaded":
            # Select framework with smallest queue
            min_queue_size = float('inf')
            target_fw_id = framework_ids[0]
            
            for fw_id in framework_ids:
                target = self.target_frameworks_map[fw_id]
                if isinstance(target, simpy.Store):
                    queue_size = len(target.items)
                    if queue_size < min_queue_size:
                        min_queue_size = queue_size
                        target_fw_id = fw_id
                        
        elif self.load_balancing_strategy == "weighted_random":
            # Weighted random (weights would need to be configured)
            weights = self.config.get("framework_weights", {})
            fw_weights = [weights.get(fw_id, 1.0) for fw_id in framework_ids]
            target_fw_id = random.choices(framework_ids, weights=fw_weights)[0]
            
        elif self.load_balancing_strategy == "session_affinity":
            # Route conversational turns to same framework
            if request.is_conversational_turn:
                # Check if we've seen this session before
                session_fw_mapping = getattr(self, '_session_fw_mapping', {})
                if request.session_id in session_fw_mapping:
                    mapped_fw_id = session_fw_mapping[request.session_id]
                    # Check if the mapped framework still exists
                    if mapped_fw_id in framework_ids:
                        target_fw_id = mapped_fw_id
                    else:
                        # Framework no longer available, select a new one
                        target_fw_id = framework_ids[self.request_counter % len(framework_ids)]
                        session_fw_mapping[request.session_id] = target_fw_id
                else:
                    # New session, use round-robin
                    target_fw_id = framework_ids[self.request_counter % len(framework_ids)]
                    session_fw_mapping[request.session_id] = target_fw_id
                    self._session_fw_mapping = session_fw_mapping
            else:
                # Non-conversational, use round-robin
                target_fw_id = framework_ids[self.request_counter % len(framework_ids)]
                if hasattr(self, '_session_fw_mapping'):
                    self._session_fw_mapping[request.session_id] = target_fw_id
                    
        else:
            # Default to round-robin
            logger.warning(f"Unknown load balancing strategy: {self.load_balancing_strategy}, using round_robin")
            target_fw_id = framework_ids[self.request_counter % len(framework_ids)]
        
        # Track request counts
        self.framework_request_counts[target_fw_id] += 1
        
        return target_fw_id