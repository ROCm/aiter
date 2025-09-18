rm mha_llama_*.log
./mha.sh 64 8 1024 1024 128     |& tee mha_llama_prefill_bf16_1024_1024.log
./mha.sh 64 8 4096 4096 128     |& tee mha_llama_prefill_bf16_4096_4096.log
./mha.sh 64 8 10240 10240 128   |& tee mha_llama_prefill_bf16_10240_10240.log
./mha.sh 64 8 1 1024 128        |& tee mha_llama_decode_bf16_1_1024.log
./mha.sh 64 8 1 4096 128        |& tee mha_llama_decode_bf16_1_4096.log
./mha.sh 64 8 1 10240 128       |& tee mha_llama_decode_bf16_1_10240.log

rm mha_qwen_*.log
./mha.sh 64 4 1024 1024 128     |& tee mha_qwen_prefill_bf16_1024_1024.log
./mha.sh 64 4 4096 4096 128     |& tee mha_qwen_prefill_bf16_4096_4096.log
./mha.sh 64 4 10240 10240 128   |& tee mha_qwen_prefill_bf16_10240_10240.log
./mha.sh 64 4 1 1024 128        |& tee mha_qwen_decode_bf16_1_1024.log
./mha.sh 64 4 1 4096 128        |& tee mha_qwen_decode_bf16_1_4096.log
./mha.sh 64 4 1 10240 128       |& tee mha_qwen_decode_bf16_1_10240.log

rm mha_deepseek_*.log
./mla.sh 128 1024 bf16 	|& tee mla_deepseek_bf16_128_1024.log
./mla.sh 128 4096 bf16 	|& tee mla_deepseek_bf16_128_4096.log
./mla.sh 128 10240 bf16 |& tee mla_deepseek_bf16_128_10240.log

grep "MHA RESULT" mha_*.log
grep "MLA RESULT" mla_*.log
