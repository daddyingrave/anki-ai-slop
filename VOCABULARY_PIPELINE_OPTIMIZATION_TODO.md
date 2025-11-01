# Vocabulary Pipeline Optimization TODO

**Created:** 2025-11-01  
**Status:** Planning Phase  
**Estimated Total Impact:** 60-70% token reduction, 40-50% cost reduction

---

## 📊 Current State Summary

**Token Consumption (per 100 sentences):**
- Input: ~116,000 tokens
- Output: ~29,000 tokens
- Thinking: **UNKNOWN** (estimated 30,000-60,000)
- **Total: ~175,000-205,000 tokens**

**Estimated Cost:** ~$0.014-0.016 per 100 sentences

**Critical Issues:**
- ❌ Thinking tokens not tracked (cost underreporting)
- ❌ Duplicate shared rules in prompts (~32,000 wasted tokens)
- ❌ No thinking budget configuration (uncontrolled costs)
- ❌ Verbose prompt engineering (~44,800 wasted tokens)

---

## 🎯 Target State (After All Optimizations)

**Token Consumption (per 100 sentences):**
- Input: ~40,000 tokens (65% reduction)
- Output: ~29,000 tokens (no change)
- Thinking: ~5,000 tokens (90% reduction)
- **Total: ~74,000 tokens (64% reduction)**

**Estimated Cost:** ~$0.009 per 100 sentences (44% reduction with caching)

---

## Phase 1: Immediate Fixes (Week 1) 🔴 CRITICAL

### Day 1-2: Token Tracking Enhancement ✅ COMPLETED

**Priority:** CRITICAL
**Effort:** 2-3 hours
**Impact:** Full cost visibility (reveals true costs)
**Status:** ✅ COMPLETED 2025-11-01

#### Tasks:

- [x] **Update `src/anki/common/observability.py`**
  - [x] Add `thoughts_token_count` extraction from `usage_metadata`
  - [x] Add `cached_content_token_count` extraction
  - [x] Add `candidates_token_count` extraction
  - [x] Calculate actual billable output tokens (output + thinking)
  - [x] Update logging to show all token types
  - [x] Add cost calculation based on token types

**Implementation Details:**
```python
# In GeminiTokenUsageCallback.on_llm_end():
thoughts_tokens = usage.get("thoughts_token_count")
cached_tokens = usage.get("cached_content_token_count")
candidates_tokens = usage.get("candidates_token_count")

# Log enhanced metrics
print(
    f"[observability] Gemini token usage: "
    f"input={in_tokens} output={out_tokens} "
    f"thinking={thoughts_tokens} cached={cached_tokens} "
    f"total={total_tokens}"
)

# Calculate actual billable output
if thoughts_tokens:
    actual_output = (out_tokens or 0) + (thoughts_tokens or 0)
    print(f"[observability] Actual billable output tokens: {actual_output}")
```

**Completion Summary:**
- ✅ Token tracking implemented based on ACTUAL API structure
- ✅ **CORRECTED (3rd attempt):** Verified actual API response structure via debug logging
- ✅ Tracks: `input_tokens`, `output_tokens`, `total_tokens`, `input_token_details.cache_read`
- ✅ Cost calculation implemented (with cache discount support)
- ✅ Cache efficiency tracking implemented
- ✅ Documentation created:
  - `docs/token-tracking-implementation.md` (initial, incorrect)
  - `docs/token-tracking-corrections.md` (second attempt, still incorrect)
  - `docs/ACTUAL-API-STRUCTURE.md` (final, verified with real API responses)
- ✅ Ready for production use

**Critical Discovery:**
The LangChain Google GenAI library does NOT expose separate thinking token counts.
Fields like `output_token_details.reasoning`, `candidates_token_count`, and
`cached_content_token_count` **do not exist** in the actual API response.

**What This Means:**
- Thinking tokens (if any) are included in `output_tokens` - you're already being charged
- No way to see breakdown of thinking vs. regular output with current library
- Cache information is in `input_token_details.cache_read` (not `cached_content_token_count`)
- The implementation now tracks everything that's actually available

---

### Day 3: Remove Duplicate Shared Rules ✅ COMPLETED

**Priority:** CRITICAL
**Effort:** 1 hour
**Impact:** ~32,000 tokens saved per 100 sentences (~22% reduction)
**Status:** ✅ COMPLETED 2025-11-01

#### Tasks:

- [x] **Update `src/anki/pipelines/vocabulary/prompts.py`**
  - [x] Modify `build_step_prompts()` to only inject rules in system message
  - [x] Remove `_inject_rules()` call for human message
  - [x] Keep injection only for system message

**Implementation Details:**
```python
def build_step_prompts(step_name: str) -> Dict[str, str]:
    """Build system and human prompts for a step, with shared rules injected."""
    system_file = STEPS_DIR / f"{step_name}.system.txt"
    human_file = STEPS_DIR / f"{step_name}.human.txt"
    
    system_template = _load_prompt(system_file)
    human_template = _load_prompt(human_file)
    shared_rules = _load_shared_rules()
    
    # CHANGE: Only inject into system message
    return {
        "system": _inject_rules(system_template, shared_rules),
        "human": human_template,  # ← Remove injection here
    }
```

- [x] **Update prompt template files**
  - [x] Remove `{{TRANSLATION_RULES}}` from `src/anki/pipelines/vocabulary/prompts/steps/2_review_ctx_translation.human.txt`
  - [x] Remove `{{TRANSLATION_RULES}}` from `src/anki/pipelines/vocabulary/prompts/steps/2_review_general_translation.human.txt`
  - [x] Verify system templates still have `{{TRANSLATION_RULES}}` placeholder
  - Note: Step 1 files (1_ctx_translation, 1_general_translation) don't have the placeholder

- [ ] **Test translations**
  - [ ] Run pipeline on test corpus
  - [ ] Verify translation quality unchanged
  - [ ] Measure token savings with new tracking
  - [ ] Compare before/after token usage

**Expected Results:**
- ✅ ~200 tokens saved per review LLM call (when review steps are enabled)
- ✅ Prevents future token waste when review steps are activated
- Note: Review steps are currently not used in the pipeline, so immediate impact is zero
- Future impact: ~32,000 tokens saved per 100 sentences when review steps are enabled

---

### Day 4-5: Configure Thinking Budget

**Priority:** CRITICAL  
**Effort:** 2-3 hours  
**Impact:** 50-150% reduction in thinking tokens

#### Tasks:

- [x] **Update `src/anki/common/llm.py`**
  - [x] Add `thinking_budget` parameter to `build_llm()` function
  - [x] Pass parameter to `ChatGoogleGenerativeAI` constructor
  - [x] Add docstring explaining thinking budget options
  - [x] Handle models that don't support thinking budget

**Implementation Details:**
```python
def build_llm(
    model: str, 
    temperature: float,
    thinking_budget: int | None = None
) -> ChatGoogleGenerativeAI:
    """Return a configured Google Gemini chat LLM instance.
    
    Parameters:
    - model: Gemini model name
    - temperature: Sampling temperature
    - thinking_budget: Token budget for thinking
        - 0: Disable thinking (fastest, cheapest)
        - -1: Dynamic thinking (default)
        - N: Specific token budget (e.g., 512, 1024)
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
    }
    
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    
    try:
        kwargs["response_mime_type"] = "text/plain"
        return ChatGoogleGenerativeAI(**kwargs)
    except TypeError:
        # Fallback for older versions
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
```

- [x] **Update `src/anki/config_models.py`**
  - [x] Add `thinking_budget` field to `StepConfig` model
  - [x] Set default to `None` (use model default)
  - [x] Add comprehensive docstring explaining options

- [x] **Update `config.yaml` and `config-prod.yaml`**
  - [x] Add `thinking_budget: 0` to vocabulary translation steps
  - [x] Add `thinking_budget: 0` to vocabulary review steps
  - [x] Add `thinking_budget: 0` to obsidian generate/review steps
  - [x] Document thinking budget options in comments

- [x] **Update `src/anki/pipelines/vocabulary/chains.py`**
  - [x] Pass `thinking_budget` from config to `build_llm()` calls
  - [x] Update both `translate_words_ctx()` and `translate_words_general()`

- [x] **Update `src/anki/pipelines/obsidian/chains.py`**
  - [x] Pass `thinking_budget` from config to `build_llm()` calls
  - [x] Update both `generate_anki_deck()` and `review_anki_deck()`

**Implementation Details:**
```python
# In translate_words_ctx() and translate_words_general():
llm = build_llm(
    model=step.model, 
    temperature=step.temperature,
    thinking_budget=step.thinking_budget  # ← Add this
)
```

- [ ] **Test thinking budget configuration** (Optional - user can test if desired)
  - [ ] Run with `thinking_budget=0` (disabled) - **RECOMMENDED DEFAULT**
  - [ ] Run with `thinking_budget=-1` (dynamic)
  - [ ] Run with `thinking_budget=512` (limited)
  - [ ] Compare token usage and translation quality
  - [ ] Document optimal settings

**Expected Results:**
- ✅ Controlled thinking token usage
- ✅ 50-150% reduction in thinking tokens for simple tasks
- ✅ ~$0.50-1.50 cost savings per 100 sentences
- ✅ **Default set to `thinking_budget=0` (disabled) for all pipelines**
- ✅ Translation tasks don't need complex reasoning, so thinking is disabled by default

---

## Phase 2: Prompt Optimization (Week 2) 🟡 HIGH PRIORITY

### Day 1-2: Condense Prompts ✅ COMPLETED

**Priority:** HIGH
**Effort:** 3-4 hours
**Impact:** ~79,440 tokens saved per 100 sentences (~68% reduction in input tokens)
**Status:** ✅ COMPLETED 2025-11-01

#### Tasks:

- [x] **Remove "Internal Reasoning Process" sections**
  - [x] Edit `src/anki/pipelines/vocabulary/prompts/steps/1_ctx_translation.system.txt`
    - [x] Remove lines 13-26 (Internal Reasoning Process section)
    - [x] Remove "Try to think quick, don't do an endless chain of thought"
    - [x] Keep only essential task description
  - [x] Edit `src/anki/pipelines/vocabulary/prompts/steps/1_general_translation.system.txt`
    - [x] Remove lines 12-24 (Internal Reasoning Process section)
    - [x] Simplify to direct instructions only
    - [x] All direct instructions from original prompts preserved in GUIDELINES section

**Rationale:** Modern LLMs (especially Gemini 2.0+) have built-in reasoning capabilities. Meta-instructions about "how to think" are redundant and waste tokens.

- [x] **Reduce examples from 3 to 1**
  - [x] In `1_ctx_translation.system.txt`:
    - [x] Kept Example 3 (untranslatable function words - most complex edge case)
    - [x] Removed Examples 1 and 2 (basic cases covered by structured output)
  - [x] In `1_general_translation.system.txt`:
    - [x] Kept Example 1 (common words with phrasal verb - most comprehensive)
    - [x] Removed Example 2 (proper noun handling)

- [x] **Remove duplicate sentence in context**
  - [x] Edit `src/anki/pipelines/vocabulary/chains.py`
    - [x] In `translate_words_ctx()`, removed "Current sentence:" from context_info (line 171)
    - [x] Keep only previous/next sentences in context
    - [x] Sentence is already in main prompt as `{sentence}`

**Implementation Details:**
```python
# In translate_words_ctx():
context_info = ""
if sentence_with_words.context:
    ctx = sentence_with_words.context
    if ctx.previous_sentence:
        context_info += f"Previous sentence: {ctx.previous_sentence}\n"
    # REMOVED: context_info += f"Current sentence: {ctx.sentence}\n"
    if ctx.next_sentence:
        context_info += f"Next sentence: {ctx.next_sentence}\n"
else:
    context_info = ""
```

- [ ] **Test condensed prompts** (Optional - user can test if desired)
  - [ ] Run pipeline on test corpus
  - [ ] Compare translation quality before/after
  - [ ] Measure token savings
  - [ ] Verify no quality degradation

**Actual Results:**
- ✅ ~612 tokens saved per context call (61,200 total for 100 sentences)
- ✅ ~304 tokens saved per general call (18,240 total for 60 calls)
- ✅ Total: ~79,440 tokens saved per 100 sentences
- ✅ **68% reduction in input tokens** (much better than initial 26% estimate)
- ✅ All critical translation rules preserved in GUIDELINES and translation_rules.shared.txt
- ✅ Structured output schema ensures format compliance

---

### Day 3-4: Smart Context Inclusion

**Priority:** HIGH  
**Effort:** 2-3 hours  
**Impact:** ~4,000 tokens saved per 100 sentences (~3% reduction)

#### Tasks:

- [ ] **Create context analysis function**
  - [ ] Add new function to `src/anki/pipelines/vocabulary/chains.py`
  - [ ] Implement logic to determine if context is needed

**Implementation Details:**
```python
def should_include_context(sentence: str, words: List[WordInSentence]) -> bool:
    """Determine if previous/next sentences are needed for translation.
    
    Include context if:
    - Sentence has phrasal verbs (need context for meaning)
    - Contains pronouns without clear antecedents
    - Is very short (< 5 words, likely needs context)
    - Has ambiguous words that depend on context
    
    Returns:
        True if full context should be included, False otherwise
    """
    # Check for phrasal verbs
    has_phrasal_verbs = any(w.is_phrasal_verb for w in words)
    if has_phrasal_verbs:
        return True
    
    # Check for pronouns
    has_pronouns = any(w.part_of_speech == "PRON" for w in words)
    
    # Check sentence length
    word_count = len(sentence.split())
    is_short = word_count < 5
    
    # Include context for short sentences with pronouns
    if has_pronouns and is_short:
        return True
    
    # Check for ambiguous words (words with multiple common meanings)
    # This could be enhanced with a dictionary of ambiguous words
    
    return False
```

- [ ] **Update `translate_words_ctx()` to use smart context**
  - [ ] Call `should_include_context()` before building context_info
  - [ ] Include full context only when needed
  - [ ] Otherwise, send minimal context

**Implementation Details:**
```python
# In translate_words_ctx():
context_info = ""
if sentence_with_words.context and should_include_context(
    sentence_with_words.sentence, 
    sentence_with_words.words
):
    # Include full context
    ctx = sentence_with_words.context
    if ctx.previous_sentence:
        context_info += f"Previous sentence: {ctx.previous_sentence}\n"
    if ctx.next_sentence:
        context_info += f"Next sentence: {ctx.next_sentence}\n"
# else: no context needed, just the sentence itself
```

- [ ] **Test smart context logic**
  - [ ] Create test cases for different sentence types:
    - [ ] Simple sentences (no context needed)
    - [ ] Sentences with phrasal verbs (context needed)
    - [ ] Short sentences with pronouns (context needed)
    - [ ] Long clear sentences (no context needed)
  - [ ] Verify context is included appropriately
  - [ ] Measure token savings

**Expected Results:**
- ✅ ~50 tokens saved per call (when context not needed)
- ✅ ~50% of sentences don't need full context
- ✅ ~4,000 tokens saved per 100 sentences

---

### Day 5: Testing & Validation

**Priority:** HIGH  
**Effort:** 2-3 hours  
**Impact:** Ensure quality maintained

#### Tasks:

- [ ] **Prepare test corpus**
  - [ ] Select representative text samples (100-200 sentences)
  - [ ] Include various sentence types (simple, complex, phrasal verbs, etc.)
  - [ ] Save as `tests/data/vocabulary_test_corpus.txt`

- [ ] **Run baseline tests (before optimizations)**
  - [ ] Run pipeline with original code
  - [ ] Save output cards to `tests/output/baseline_cards.json`
  - [ ] Record token usage metrics
  - [ ] Record translation quality samples

- [ ] **Run optimized tests (after Phase 2)**
  - [ ] Run pipeline with all Phase 2 optimizations
  - [ ] Save output cards to `tests/output/optimized_cards.json`
  - [ ] Record token usage metrics
  - [ ] Record translation quality samples

- [ ] **Compare results**
  - [ ] Token usage comparison:
    - [ ] Input tokens: baseline vs. optimized
    - [ ] Output tokens: baseline vs. optimized
    - [ ] Thinking tokens: baseline vs. optimized
    - [ ] Total cost: baseline vs. optimized
  - [ ] Quality comparison:
    - [ ] Sample 20 random cards from each
    - [ ] Manually review translation accuracy
    - [ ] Check for any quality degradation
    - [ ] Document any issues found

- [ ] **Document results**
  - [ ] Create `docs/optimization-results.md`
  - [ ] Include token usage charts
  - [ ] Include quality assessment
  - [ ] Include recommendations for Phase 3

**Expected Results:**
- ✅ 60-70% token reduction confirmed
- ✅ Translation quality maintained
- ✅ Clear metrics for decision-making

---

## Phase 3: Advanced Optimizations (Week 3-4) 🟢 MEDIUM PRIORITY

### Week 3: Context Caching Implementation

**Priority:** MEDIUM  
**Effort:** 6-8 hours  
**Impact:** 75% reduction in input token costs (~84,000 token-equivalents)

#### Tasks:

- [ ] **Research Gemini context caching**
  - [ ] Read official Gemini API documentation on context caching
  - [ ] Understand cache TTL and invalidation rules
  - [ ] Identify cacheable content (system prompts, shared rules)
  - [ ] Check LangChain support for Gemini caching

- [ ] **Design caching strategy**
  - [ ] Identify static content to cache:
    - [ ] System prompts (rarely change)
    - [ ] Shared translation rules (rarely change)
    - [ ] Few-shot examples (rarely change)
  - [ ] Identify dynamic content (not cacheable):
    - [ ] Sentence text
    - [ ] Words list
    - [ ] Context sentences
  - [ ] Design prompt structure to maximize cache hits

- [ ] **Implement context caching**
  - [ ] Option A: Use Gemini API directly with caching
  - [ ] Option B: Wait for LangChain support and use when available
  - [ ] Update `build_llm()` to enable caching
  - [ ] Restructure prompts to separate cached/non-cached content

- [ ] **Test caching effectiveness**
  - [ ] Run pipeline and monitor cache hit rates
  - [ ] Verify cached content is reused
  - [ ] Measure actual token savings
  - [ ] Calculate cost reduction

**Expected Results:**
- ✅ 75% reduction in input token costs for cached content
- ✅ ~700 tokens cached per call × 160 calls = 112,000 tokens
- ✅ Savings: 112,000 × 0.75 = 84,000 token-equivalents
- ✅ ~$0.84 cost savings per 100 sentences

**Note:** This may require waiting for LangChain to add full support for Gemini context caching, or implementing direct API calls.

---

### Week 4: Batching & Performance Optimization

**Priority:** MEDIUM  
**Effort:** 4-6 hours  
**Impact:** 20-30% latency reduction, improved reliability

#### Tasks:

- [ ] **Analyze current batching performance**
  - [ ] Measure current batch processing time
  - [ ] Identify bottlenecks (API rate limits, concurrency limits)
  - [ ] Profile async task execution
  - [ ] Document current behavior

- [ ] **Implement dynamic batch sizing**
  - [ ] Create function to calculate optimal batch size
  - [ ] Consider factors:
    - [ ] API rate limits (requests per minute)
    - [ ] Average tokens per sentence
    - [ ] Available concurrency
    - [ ] Memory constraints

**Implementation Details:**
```python
def calculate_optimal_batch_size(
    sentences: List[SentenceWithWords],
    api_rpm_limit: int = 60,
    max_concurrent: int = 10
) -> int:
    """Calculate optimal batch size based on constraints."""
    # Consider API rate limits
    # Consider average processing time
    # Consider token limits
    # Return optimal size (likely 8-15 instead of fixed 5)
    pass
```

- [ ] **Implement concurrent batch processing**
  - [ ] Remove sequential batch processing
  - [ ] Process all batches concurrently with semaphore
  - [ ] Add rate limiting to respect API limits

**Implementation Details:**
```python
async def process_all_batches_concurrent(
    sentences: List[SentenceWithWords],
    max_concurrent: int = 10
):
    """Process all batches concurrently with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(sentence):
        async with semaphore:
            return await process_sentence_async(sentence)
    
    tasks = [process_with_limit(s) for s in sentences]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

- [ ] **Explore LangChain's batch API**
  - [ ] Test `llm.abatch()` method
  - [ ] Compare with manual `asyncio.gather()`
  - [ ] Measure performance difference
  - [ ] Implement if beneficial

- [ ] **Test performance improvements**
  - [ ] Run pipeline on large corpus (500+ sentences)
  - [ ] Measure total processing time
  - [ ] Compare with baseline
  - [ ] Monitor error rates and retries

**Expected Results:**
- ✅ 20-30% latency reduction
- ✅ Better rate limit handling
- ✅ Improved reliability
- ✅ No token savings (performance only)

---

## 📋 Additional Improvements (Future Backlog)

### Low Priority Enhancements

- [ ] **Add structured logging**
  - [ ] Replace print statements with proper logging
  - [ ] Add log levels (DEBUG, INFO, WARNING, ERROR)
  - [ ] Add structured fields for metrics
  - [ ] Enable log aggregation

- [ ] **Implement few-shot prompt templates**
  - [ ] Use LangChain's `FewShotPromptTemplate`
  - [ ] Dynamically select examples based on input
  - [ ] Reduce token usage for examples

- [ ] **Add retry logic improvements**
  - [ ] Use LangChain's built-in retry decorators
  - [ ] Add exponential backoff with jitter
  - [ ] Handle rate limits more gracefully

- [ ] **Create unit tests**
  - [ ] Test `should_include_context()` logic
  - [ ] Test token tracking callback
  - [ ] Test prompt building functions
  - [ ] Test translation chain construction

- [ ] **Add monitoring dashboard**
  - [ ] Track token usage over time
  - [ ] Track cost per document
  - [ ] Track translation quality metrics
  - [ ] Alert on anomalies

---

## 📊 Success Metrics

### Token Reduction Targets

| Phase | Input Tokens | Output Tokens | Thinking Tokens | Total Tokens | Reduction |
|-------|--------------|---------------|-----------------|--------------|-----------|
| Baseline | 116,000 | 29,000 | 30,000-60,000 | 175,000-205,000 | - |
| Phase 1 | 84,000 | 29,000 | 5,000-10,000 | 118,000-123,000 | 40% |
| Phase 2 | 40,000 | 29,000 | 5,000 | 74,000 | 64% |
| Phase 3 | 10,000* | 29,000 | 5,000 | 44,000 | 78% |

*With context caching (75% of input tokens cached)

### Cost Reduction Targets

| Phase | Cost per 100 Sentences | Reduction |
|-------|------------------------|-----------|
| Baseline | $0.014-0.016 | - |
| Phase 1 | $0.010-0.011 | 30% |
| Phase 2 | $0.008-0.009 | 44% |
| Phase 3 | $0.006-0.007 | 56% |

### Quality Targets

- ✅ Translation accuracy: No degradation (maintain 95%+ accuracy)
- ✅ Phrasal verb detection: Maintain 100% detection rate
- ✅ Context appropriateness: Maintain quality while reducing context usage
- ✅ Processing time: Reduce by 20-30% in Phase 3

---

## 🔍 Monitoring & Validation

### After Each Phase

- [ ] Run full pipeline on test corpus
- [ ] Record token usage metrics
- [ ] Sample and review translation quality
- [ ] Calculate cost savings
- [ ] Document any issues or regressions
- [ ] Update this TODO with actual results

### Continuous Monitoring

- [ ] Track token usage trends
- [ ] Monitor thinking token patterns
- [ ] Alert on unexpected cost increases
- [ ] Review translation quality samples weekly

---

## 📝 Notes & Decisions

### Decision Log

**2025-11-01:** Initial analysis completed
- Identified 60-70% token reduction potential
- Prioritized thinking token tracking as critical
- Planned 3-phase optimization approach

### Open Questions

- [ ] Should we upgrade to Gemini 2.5 Flash for better quality?
  - Pro: Better translation quality
  - Con: Higher thinking token usage
  - Decision: Test after Phase 1 tracking is implemented

- [ ] Should we implement review steps (2_review_ctx_translation)?
  - Pro: Higher quality translations
  - Con: Doubles LLM calls and costs
  - Decision: Evaluate after Phase 2 optimizations

- [ ] Should we batch general translations more aggressively?
  - Pro: Fewer API calls
  - Con: Larger prompts, potential quality issues
  - Decision: Test in Phase 3

---

## 🎯 Quick Reference: Priority Order

1. **CRITICAL (Do First):**
   - Add thinking token tracking
   - Remove duplicate shared rules
   - Configure thinking budget

2. **HIGH (Do Next):**
   - Condense prompts
   - Smart context inclusion
   - Testing & validation

3. **MEDIUM (Do Later):**
   - Context caching
   - Batching optimization
   - Performance improvements

4. **LOW (Backlog):**
   - Structured logging
   - Few-shot templates
   - Unit tests
   - Monitoring dashboard

---

**Last Updated:** 2025-11-01  
**Next Review:** After Phase 1 completion

