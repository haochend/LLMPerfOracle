Document 8: Simulating Prefix Caching
1. Introduction
* Purpose: This document details the design for simulating prefix caching (also known as KV cache reuse or context caching) within the virtual LLM serving testing environment.
* Benefit: Prefix caching is a critical optimization in LLM serving. It significantly reduces the computational cost of the prefill phase for requests that share common leading tokens with previous requests (e.g., conversational history, shared system prompts, common instruction prefixes). Simulating this feature accurately is crucial for realistic performance evaluation and understanding its benefits under different conditions.
* Core Idea: The simulation will involve framework modules identifying shared prefixes between incoming requests and already processed data. For the cached portion of the prompt, the prefill computation will be skipped or significantly reduced, and only the new, unique tokens will be prefilled. The existing KV cache entries for the prefix are reused.
2. Prerequisites & Existing Support
* PagedAttention/KV Cache Management (Document 4): The foundational mechanism for managing KV cache blocks (e.g., gpu_blocks_container in the VLLMFramework example) is assumed to be in place. This infrastructure allows for the storage and potential reuse of KV cache entries.
* Conversational Workloads (Document 3): The WorkloadGenerator is capable of producing sequences of requests linked by a session_id and marked with is_conversational_turn. These workloads are primary scenarios for prefix caching.
* Metrics Collection (Document 5): The MetricsCollector will be enhanced to log and report on prefix caching effectiveness.
3. Modifications to Framework Modules (e.g., VLLMFramework)
The core logic for prefix caching simulation resides within the concrete LLMFrameworkModule implementations.
* 3.1. Tracking Active Sequences and Their KV Cache State:
   * The SequenceState data class (defined in/for Document 4) needs to store information relevant to the prefix that has been processed and whose KV cache is available.
      * SequenceState.prompt_tokens_fully_processed: int: Total number of prompt tokens for which KV cache has been computed and is currently "active" for this sequence.
      * SequenceState.kv_block_ids_for_prompt: List[BlockID]: (Optional, for detailed block-level simulation of reuse) A list of identifiers for the KV cache blocks storing the prompt_tokens_fully_processed.
   * active_sessions_kv_state: Dict[str, SessionCacheInfo]: A dictionary within the framework module to store information about the KV cache state of the last completed or ongoing turn for each active session_id.
      * SessionCacheInfo (Data Class):
         * session_id: str
         * total_tokens_in_cache: int (Length of the prefix + generated response currently in KV for this session)
         * prompt_part_length: int (Length of the prompt part of total_tokens_in_cache)
         * response_part_length: int (Length of the response part of total_tokens_in_cache)
         * associated_sequence_id: str (The request_id of the sequence currently holding this session's cache)
         * kv_block_ids: List[BlockID] (Optional, for detailed block management)
         * last_update_time: float (Simulation time, for potential eviction policies)
* 3.2. Enhanced Request Admission/Prefill Scheduling Logic:
This logic is typically part of the framework module's main processing_loop or a dedicated request intake method.
   1. Initialize num_tokens_to_prefill_this_turn and cached_prefix_length_for_this_request:
   * num_tokens_to_prefill_this_turn = current_request.prompt_num_tokens
   * cached_prefix_length_for_this_request = 0
   2. Check for Conversational Prefix (Primary Cache Hit Scenario):
   * If current_request.is_conversational_turn and current_request.session_id is in self.active_sessions_kv_state:
   * session_cache_info = self.active_sessions_kv_state[current_request.session_id]
   * Assume the new prompt current_request.prompt_num_tokens is constructed as (history + previous_response) + new_user_query. The (history + previous_response) part corresponds to session_cache_info.total_tokens_in_cache.
   * If current_request.prompt_num_tokens > session_cache_info.total_tokens_in_cache:
   * The prompt effectively starts with the entire cached context of the session.
   * cached_prefix_length_for_this_request = session_cache_info.total_tokens_in_cache
   * num_tokens_to_prefill_this_turn = current_request.prompt_num_tokens - cached_prefix_length_for_this_request
   * The SequenceState for current_request should conceptually "attach" to the KV cache blocks represented by session_cache_info. No new blocks are allocated for the cached_prefix_length_for_this_request part.
   * Log: self.metrics_collector.log_prefix_cache_event(current_request.request_id, self.simpy_env.now, "CONVERSATIONAL_HIT", cached_prefix_length_for_this_request)
   * Else (e.g., prompt is shorter or unrelated, though this shouldn't happen if is_conversational_turn is true and prompts are built correctly):
   * Treat as a cache miss for this specific logic.
   * Log: self.metrics_collector.log_prefix_cache_event(current_request.request_id, self.simpy_env.now, "CONVERSATIONAL_MISS_UNEXPECTED_PROMPT", 0)
   3. (Advanced/Optional) Cross-Request Prefix Caching (e.g., for shared system prompts):
   * Requires a global prefix data structure: global_prefix_store: Dict[PrefixHash, GlobalPrefixCacheInfo]
   * PrefixHash: Hash of the token sequence of the prefix.
   * GlobalPrefixCacheInfo: length, kv_block_ids, reference_count, last_access_time.
   * If no conversational hit, or as a secondary check:
   * Iterate through potential prefix lengths of current_request.prompt_tokens (e.g., from max common prefix length down to a minimum).
   * For each potential prefix, check if its hash exists in global_prefix_store.
   * If a match is found:
   * cached_prefix_info = global_prefix_store[matched_prefix_hash]
   * cached_prefix_length_for_this_request = cached_prefix_info.length
   * num_tokens_to_prefill_this_turn = current_request.prompt_num_tokens - cached_prefix_length_for_this_request
   * Increment cached_prefix_info.reference_count.
   * Log: self.metrics_collector.log_prefix_cache_event(current_request.request_id, self.simpy_env.now, "CROSS_REQUEST_HIT", cached_prefix_length_for_this_request)
   * Break from prefix search.
   * If no hit after all checks:
   * Log: self.metrics_collector.log_prefix_cache_event(current_request.request_id, self.simpy_env.now, "MISS_FULL", 0)
   4. Update SequenceState for the current_request:
   * current_sequence_state.cached_prefix_length_used = cached_prefix_length_for_this_request
   * current_sequence_state.num_tokens_requiring_prefill = num_tokens_to_prefill_this_turn
   * 3.3. Modified Prefill Computation Task:
   * The call to _estimate_prefill_ops and the subsequent submission to virtual_hardware.submit_computation_task must use current_sequence_state.num_tokens_requiring_prefill.
# In VLLMFramework.processing_loop, when preparing for prefill:
# ... (logic from 3.2 to determine num_tokens_requiring_prefill for current_sequence_state) ...

if current_sequence_state.num_tokens_requiring_prefill > 0:
   task_desc_prefill = self._estimate_prefill_ops(
       self.model_profile,
       current_sequence_state.num_tokens_requiring_prefill, # KEY CHANGE
       batch_size=1 # Or actual batch size if batching prefills
   )
   # ... (set task_id, log prefill start with actual tokens being prefilled)
   self.metrics_collector.log_prefill_start(
       current_sequence_state.request.request_id, 
       self.simpy_env.now, 
       current_sequence_state.num_tokens_requiring_prefill # Log actual tokens prefilled
   )
   # ... yield self.simpy_env.process(self.virtual_hardware.submit_computation_task(...)) ...
   current_sequence_state.prompt_tokens_fully_processed = current_sequence_state.cached_prefix_length_used + current_sequence_state.num_tokens_requiring_prefill
else: # num_tokens_requiring_prefill is 0, full prefix cache hit
   self.metrics_collector.log_prefix_cache_event(current_sequence_state.request.request_id, self.simpy_env.now, "FULL_HIT_NO_PREFILL_NEEDED", current_sequence_state.cached_prefix_length_used)
   current_sequence_state.prompt_tokens_fully_processed = current_sequence_state.cached_prefix_length_used
   current_sequence_state.status = 'DECODING' # Transition directly to decode
   # For TTFT calculation, the "prefill phase" effectively ended now.
   # Ensure metrics_collector.log_first_token_generated uses an appropriate start time.
   # This could be the request arrival time or the time this cache hit was determined.
   # A specific log_event might be needed: log_prefill_skipped_due_to_cache_hit
   # The first token will be generated by the first *decode* step.
   # Record a "conceptual prefill end time" for accurate TTFT.
   current_sequence_state.prefill_end_time_sim = self.simpy_env.now 

   * 3.4. Updating Session and Global Prefix Caches:
   * On Conversational Turn Completion:
   * completed_sequence_state = ...
   * session_id = completed_sequence_state.request.session_id
   * new_session_cache_info = SessionCacheInfo(
   * session_id = session_id,
   * total_tokens_in_cache = completed_sequence_state.prompt_tokens_fully_processed + completed_sequence_state.output_tokens_generated,
   * prompt_part_length = completed_sequence_state.prompt_tokens_fully_processed,
   * response_part_length = completed_sequence_state.output_tokens_generated,
   * associated_sequence_id = completed_sequence_state.request.request_id,
   * last_update_time = self.simpy_env.now
   * )
   * self.active_sessions_kv_state[session_id] = new_session_cache_info
   * On Sequence Eviction / Completion (for Global Prefix Cache with reference counting):
   * If the evicted/completed sequence used a prefix from global_prefix_store:
   * Decrement reference count for that prefix.
   * If reference count becomes zero, the KV blocks for that global prefix can be freed (consider LRU eviction from global_prefix_store itself if it has a size limit).
   * Adding New Prefixes to Global Cache:
   * If a sufficiently long new prompt segment (that wasn't a cache hit) is prefilled, it could be a candidate for adding to global_prefix_store. This requires defining criteria (min length, frequency if tracked) and managing the global cache size.
   * 3.5. KV Cache Block Management for Shared Prefixes:
   * Simplified Model (Initial): Focus on reduced computation. Assume that if computation for a prefix is skipped, the memory is "logically" reused. The gpu_blocks_container tracks total unique blocks in use. This means a shared prefix doesn't double-count against the container if it's already provisioned for another active sequence that "owns" it (e.g., the original sequence in active_sessions_kv_state). This is an abstraction.
   * Detailed Model (Advanced): Implement reference counting for KV cache blocks.
   * When a prefix is reused, the new sequence shares block IDs.
   * gpu_blocks_container.get(num_new_blocks_needed_for_delta_prefill).
   * _release_sequence_resources would only decrement reference counts for shared blocks. A block is returned to gpu_blocks_container.put() only when its reference count is zero. This makes block accounting more complex but more accurate for memory footprint analysis.
4. Simulating Different Workload Types for Prefix Caching
The WorkloadGenerator (Doc 3) configuration is key to creating scenarios that test prefix caching.
   * 4.1. Conversational Workloads (High Conversational Hit Rate):
   * Workload Config: High conversational_probability, is_conversational_turn=True for follow-ups. Prompts for follow-up turns should be structured by the workload generator as full_history_plus_new_input or the framework module must be aware of how to reconstruct this from new_input_only and active_sessions_kv_state.
   * Expected Behavior: High CONVERSATIONAL_HIT rates. Significantly lower num_tokens_requiring_prefill for follow-up turns. Lower average TTFT.
   * Test Variations: Vary length of new user input in follow-up turns.
   * 4.2. Repetitive Task Workloads (High Cross-Request Hit Rate):
   * Workload Config: Generate prompts with a large, fixed leading instruction set (e.g., "You are a helpful assistant. Translate the following text to French: {variable_text}").
   * Framework Req: Requires "Cross-Request Prefix Caching" (Section 3.2, Point 3).
   * Expected Behavior: After the first few unique instruction sets are processed, subsequent requests using the same instructions should get CROSS_REQUEST_HIT.
   * Test Variations: Number of unique fixed instruction prefixes, length of variable text.
   * 4.3. Highly Diverse, Unique Prompts (Low Hit Rate):
   * Workload Config: Low conversational_probability. Prompts are mostly unique random strings/sequences.
   * Expected Behavior: High MISS_FULL rates. Performance similar to no prefix caching. num_tokens_requiring_prefill close to prompt_num_tokens.
   * 4.4. Workloads with a Limited Set of System Prompts / Few-Shot Examples:
   * Workload Config: Prompts constructed as random.choice(FEW_SHOT_PREFIX_LIST) + user_query.
   * Framework Req: Cross-Request Prefix Caching.
   * Expected Behavior: The FEW_SHOT_PREFIX_LIST items become cached. High CROSS_REQUEST_HIT rate for these.
   * Test Variations: Size of FEW_SHOT_PREFIX_LIST, length of prefixes, length of user query.
   * 4.5. "Context Window Exceeded" Scenarios for Conversational Workloads:
   * Workload Config: Long conversations where current_request.prompt_num_tokens (full history + new input) eventually exceeds the model's context window or the KV cache capacity allocated for a session.
   * Framework Behavior: The framework module would need logic to handle this. It might:
   * Truncate the oldest part of the history (sliding window).
   * The cached_prefix_length_for_this_request would then be based on the truncated history that fits.
   * This means part of the history is "forgotten" and needs to be re-prefilled if it were to be included again.
   * Expected Behavior: Prefix caching hit is partial or effectively a miss for the oldest tokens if truncation occurs.
5. Metrics Collector (Doc 5) Enhancements
   * New Logging Methods/Fields:
   * In log_prefill_start(self, request_id: str, prefill_start_time: float, num_tokens_actually_prefilled: int): Add num_tokens_actually_prefilled.
   * log_prefix_cache_event(self, request_id: str, timestamp: float, event_type: str, cached_prefix_length: int, num_tokens_prefilled_after_cache_check: int):
   * event_type: "CONVERSATIONAL_HIT", "CROSS_REQUEST_HIT", "MISS_FULL", "FULL_HIT_NO_PREFILL_NEEDED", "CONVERSATIONAL_MISS_UNEXPECTED_PROMPT".
   * New Derived Metrics in RequestMetricsEntry:
   * cached_prefix_length_used: int
   * num_tokens_actually_prefilled: int
   * New Summary Report Metrics:
   * Overall prefix cache hit rate (count of HITs / total requests eligible).
   * Conversational prefix cache hit rate.
   * Cross-request prefix cache hit rate (if implemented).
   * Average cached_prefix_length_used for hits.
   * Average num_tokens_actually_prefilled across all requests.
   * Ratio: sum(num_tokens_actually_prefilled) / sum(prompt_num_tokens). This shows overall prefill reduction.
   * Impact on TTFT: Compare TTFT for cache hits vs. misses.
6. Implementation Phases and Considerations
   1. Phase 1: Core Conversational Prefix Caching:
   * Implement active_sessions_kv_state and the logic in Section 3.2, Point 2.
   * Modify prefill computation as in Section 3.3.
   * Update active_sessions_kv_state on turn completion (Section 3.4).
   * Focus on simulating reduced computation. The "Simplified Model" for KV block management (Section 3.5) is acceptable initially.
   * Enhance metrics for basic hit/miss logging.
   2. Phase 2: Advanced Cross-Request Prefix Caching (If prioritized):
   * Implement global_prefix_store and associated logic.
   * Requires careful thought on prefix hashing/identification and cache eviction policies for the global store.
   3. Phase 3: Detailed KV Block Reference Counting (If memory accuracy is paramount):
   * Implement reference counting for shared KV blocks to accurately model memory footprint under heavy prefix sharing. This significantly increases the complexity of KV block management in the framework module and VirtualHardwarePlatform interactions.
This detailed design provides a roadmap for simulating prefix caching, enabling the virtual environment to test and quantify its performance impact under diverse and realistic workload conditions.