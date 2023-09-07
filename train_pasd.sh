TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch \
	train_pasd.py \
	--dataset_name="pasd" \
 	--pretrained_model_name_or_path="checkpoints/stable-diffusion-v1-5" \
	--output_dir="runs/pasd" \
	--resolution=512 \
	--learning_rate=5e-5 \
	--gradient_accumulation_steps=2 \
	--train_batch_size=4 \
	--num_train_epochs=1000 \
	--tracker_project_name="pasd" \
	--enable_xformers_memory_efficient_attention \
	--checkpointing_steps=10000 \
	--control_type="realisr" \
	--mixed_precision="fp16" \
	--dataloader_num_workers=64 \
#    --multi_gpu --num_processes=8 --gpu_ids '0,1,2,3,4,5,6,7' \

