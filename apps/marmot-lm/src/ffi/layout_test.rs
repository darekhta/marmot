//! Test struct layout compatibility between Rust and C

use std::mem::{size_of, offset_of};
use super::bindings::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_options_layout() {
        println!("marmot_llm_generate_options_t:");
        println!("  size: {}", size_of::<marmot_llm_generate_options_t>());
        println!("  struct_size offset: {}", offset_of!(marmot_llm_generate_options_t, struct_size));
        println!("  struct_version offset: {}", offset_of!(marmot_llm_generate_options_t, struct_version));
        println!("  flags offset: {}", offset_of!(marmot_llm_generate_options_t, flags));
        println!("  max_new_tokens offset: {}", offset_of!(marmot_llm_generate_options_t, max_new_tokens));
        println!("  stop_on_eos offset: {}", offset_of!(marmot_llm_generate_options_t, stop_on_eos));
        println!("  stop_tokens offset: {}", offset_of!(marmot_llm_generate_options_t, stop_tokens));
        println!("  stop_tokens_len offset: {}", offset_of!(marmot_llm_generate_options_t, stop_tokens_len));
        println!("  prefill_progress offset: {}", offset_of!(marmot_llm_generate_options_t, prefill_progress));
        println!("  on_token offset: {}", offset_of!(marmot_llm_generate_options_t, on_token));
        println!("  user_data offset: {}", offset_of!(marmot_llm_generate_options_t, user_data));
        println!("  pnext offset: {}", offset_of!(marmot_llm_generate_options_t, pnext));
        println!("  reserved offset: {}", offset_of!(marmot_llm_generate_options_t, reserved));

        // Expected C layout on 64-bit:
        // struct_size: 0
        // struct_version: 4
        // flags: 8
        // max_new_tokens: 16
        // stop_on_eos: 24
        // stop_tokens: 32 (after 7 bytes padding)
        // stop_tokens_len: 40
        // prefill_progress: 48
        // on_token: 56
        // user_data: 64
        // pnext: 72
        // reserved: 80
        // total size: 112

        assert_eq!(size_of::<marmot_llm_generate_options_t>(), 112, "size mismatch");
        assert_eq!(offset_of!(marmot_llm_generate_options_t, struct_size), 0);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, struct_version), 4);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, flags), 8);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, max_new_tokens), 16);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, stop_on_eos), 24);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, stop_tokens), 32);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, stop_tokens_len), 40);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, prefill_progress), 48);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, on_token), 56);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, user_data), 64);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, pnext), 72);
        assert_eq!(offset_of!(marmot_llm_generate_options_t, reserved), 80);
    }

    #[test]
    fn test_sampling_options_layout() {
        println!("\nmarmot_llm_sampling_options_t:");
        println!("  size: {}", size_of::<marmot_llm_sampling_options_t>());
        println!("  struct_size offset: {}", offset_of!(marmot_llm_sampling_options_t, struct_size));
        println!("  struct_version offset: {}", offset_of!(marmot_llm_sampling_options_t, struct_version));
        println!("  flags offset: {}", offset_of!(marmot_llm_sampling_options_t, flags));
        println!("  seed offset: {}", offset_of!(marmot_llm_sampling_options_t, seed));
        println!("  temperature offset: {}", offset_of!(marmot_llm_sampling_options_t, temperature));
        println!("  top_k offset: {}", offset_of!(marmot_llm_sampling_options_t, top_k));
        println!("  top_p offset: {}", offset_of!(marmot_llm_sampling_options_t, top_p));
        println!("  min_p offset: {}", offset_of!(marmot_llm_sampling_options_t, min_p));
        println!("  repetition_penalty offset: {}", offset_of!(marmot_llm_sampling_options_t, repetition_penalty));
        println!("  pnext offset: {}", offset_of!(marmot_llm_sampling_options_t, pnext));
        println!("  reserved offset: {}", offset_of!(marmot_llm_sampling_options_t, reserved));

        // Expected C layout on 64-bit:
        // struct_size: 0
        // struct_version: 4
        // flags: 8
        // seed: 16
        // temperature: 24
        // top_k: 32 (after 4 bytes padding)
        // top_p: 40
        // min_p: 44
        // repetition_penalty: 48
        // pnext: 56 (after 4 bytes padding)
        // reserved: 64
        // total size: 96

        assert_eq!(size_of::<marmot_llm_sampling_options_t>(), 96, "size mismatch");
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, struct_size), 0);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, struct_version), 4);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, flags), 8);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, seed), 16);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, temperature), 24);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, top_k), 32);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, top_p), 40);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, min_p), 44);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, repetition_penalty), 48);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, pnext), 56);
        assert_eq!(offset_of!(marmot_llm_sampling_options_t, reserved), 64);
    }

    #[test]
    fn test_serving_engine_options_layout() {
        println!("\nmarmot_serving_engine_options_t:");
        println!("  size: {}", size_of::<marmot_serving_engine_options_t>());
        println!("  struct_size offset: {}", offset_of!(marmot_serving_engine_options_t, struct_size));
        println!("  struct_version offset: {}", offset_of!(marmot_serving_engine_options_t, struct_version));
        println!("  flags offset: {}", offset_of!(marmot_serving_engine_options_t, flags));
        println!("  max_seqs offset: {}", offset_of!(marmot_serving_engine_options_t, max_seqs));
        println!("  max_batch_seqs offset: {}", offset_of!(marmot_serving_engine_options_t, max_batch_seqs));
        println!("  max_num_tokens offset: {}", offset_of!(marmot_serving_engine_options_t, max_num_tokens));
        println!("  max_seq_len offset: {}", offset_of!(marmot_serving_engine_options_t, max_seq_len));
        println!("  block_size offset: {}", offset_of!(marmot_serving_engine_options_t, block_size));
        println!("  num_kv_blocks offset: {}", offset_of!(marmot_serving_engine_options_t, num_kv_blocks));
        println!("  num_swap_blocks offset: {}", offset_of!(marmot_serving_engine_options_t, num_swap_blocks));
        println!("  kv_dtype offset: {}", offset_of!(marmot_serving_engine_options_t, kv_dtype));
        println!("  kv_block_watermark offset: {}", offset_of!(marmot_serving_engine_options_t, kv_block_watermark));
        println!("  prefill_chunk_size offset: {}", offset_of!(marmot_serving_engine_options_t, prefill_chunk_size));
        println!("  pnext offset: {}", offset_of!(marmot_serving_engine_options_t, pnext));
        println!("  reserved offset: {}", offset_of!(marmot_serving_engine_options_t, reserved));

        assert_eq!(size_of::<marmot_serving_engine_options_t>(), 128, "size mismatch");
    }

    #[test]
    fn test_request_state_enum_size() {
        println!("\nmarmot_llm_request_state_t:");
        println!("  size: {}", size_of::<marmot_llm_request_state_t>());

        // In C, an enum is typically 4 bytes (int)
        assert_eq!(size_of::<marmot_llm_request_state_t>(), 4, "enum size should be 4 bytes like C int");
    }
}
