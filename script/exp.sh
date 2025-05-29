export CUDA_VISIBLE_DEVICES=0
# python src/exp.py \
#     --exp common \
#     --model local \
#     --prompt_method cot

# python src/exp.py \
#     --exp common \
#     --model google \
#     --prompt_method cot

python src/exp.py \
    --exp taskc \
    --model local \
    --prompt_method multichoice

# python src/exp.py \
#     --exp taskc \
#     --model local \
#     --prompt_method cot

python src/exp.py \
    --exp taskc \
    --model google \
    --prompt_method multichoice

python src/exp.py \
    --exp taskc \
    --model google \
    --prompt_method cot
