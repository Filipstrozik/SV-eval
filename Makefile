run:
	for config in ecapa campplus ecapa2 redimnet; do \
		python src/main.py --model_config=$$config; \
	done

