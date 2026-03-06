#!/bin/bash

dataset="mnist"
epochs=3

# --- 🚀 START THE SWEEP ---
for optimizer in sgd momentum nag rmsprop
do
    for activation in relu tanh
    do
        for lr in 0.001 0.0005
        do
            for nhl in 1 2 3
            do
                # Dynamically set the hidden sizes based on the number of layers
                if [ $nhl -eq 1 ]; then
                    sizes="128"
                elif [ $nhl -eq 2 ]; then
                    sizes="128 64"
                elif [ $nhl -eq 3 ]; then
                    sizes="128 64 32"
                fi

                echo "================================================================="
                echo "Running: Opt=$optimizer, Act=$activation, LR=$lr, Layers=$nhl"
                echo "================================================================="

                # Notice we are passing the exact same --model_path every time!
                python -m src.train \
                    -d $dataset \
                    -e $epochs \
                    -b 64 \
                    -l cross_entropy \
                    -o $optimizer \
                    -lr $lr \
                    -wd 0 \
                    -nhl $nhl \
                    -sz $sizes \
                    -a $activation \
                    -w_i xavier \
                    -w_p da6401_sweep_5 \
                    --model_path src/best_model.npy

            done
        done
    done
done

echo "================================================================="
echo "🎉 Sweep complete! The absolute best model is saved at src/best_model.npy"
echo "================================================================="