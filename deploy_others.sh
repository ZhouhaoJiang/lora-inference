#export MODEL_ID="jzli/majicMIX-realistic-7" # change this
export MODEL_ID="Lykon/dreamshaper-8" # change this
export SAFETY_MODEL_ID="CompVis/stable-diffusion-safety-checker"
export IS_FP16=1
export USERNAME="zhouhaojiang" # change this
#export REPLICATE_MODEL_ID="majicmix_with_lora" # change this
export REPLICATE_MODEL_ID="dreamshaper_8_with_lora" # change this

echo "MODEL_ID=$MODEL_ID" > .env
echo "SAFETY_MODEL_ID=$SAFETY_MODEL_ID" >> .env
echo "IS_FP16=$IS_FP16" >> .env

cog run script/download-weights.py
cog run python test.py --test_img2img --test_text2img --test_adapter
cog push r8.im/$USERNAME/$REPLICATE_MODEL_ID
