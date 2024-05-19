import os
from generate import *
from tokenizer import *
import errno
import tempfile 

def process_midi(model_, inp, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None,
             tempo=512820, verbose=False, token_count=None):
    
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(inp)
        fp.close()
        fp_name = fp.name
        print("Processing midi file")
        midi_parser_output = midi_parser(fp.name)
        tempo = midi_parser_output[2]
        midi_input = midi_parser_output[1]
        
        os.unlink(fp_name)

        print(midi_input)

        generate(model_=model_, inp=midi_input, save_path=save_path,
                 temperature=temperature, mode=mode, k=k, tempo=tempo, verbose=verbose, token_count=token_count)



if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(
        prog="generate_realtime.py",
        description="Generate midi audio real time "
    )
    parser.add_argument("path_to_model", help="string path to a .pt file at which has been saved a dictionary "
                                              "containing the model state dict and hyperparameters", type=str)
    parser.add_argument("save_path", help="path at which to save the generated midi file", type=str)
    
    parser.add_argument("-c", "--compile", help="if true, model will be `torch.compile`d for potentially better "
                                                "speed; default: false", action="store_true")
    parser.add_argument("-m", "--mode", help="specify 'categorical' or 'argmax' mode of decode sampling", type=str)
    parser.add_argument("-k", "--top-k", help="top k samples to consider while decode sampling; default: all",
                        type=check_positive)
    parser.add_argument("-t", "--temperature",
                        help="temperature for decode sampling; lower temperature, more sure the sampling, "
                             "higher temperature, more diverse the output; default: 1.0 (categorical sample of true "
                             "model output)",
                        type=float)
    parser.add_argument("-tm", "--tempo", help="approximate tempo of generated sample in BMP", type=check_positive)
    parser.add_argument("-n", "--number", help="Number of tokens to generate", type=int)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")
    parser.add_argument("-p", "--pipe_name", help="Pipe name", type=str)

    args = parser.parse_args()

    # fix arguments
    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else 512820
    token_count = args.number if args.number else None
    pipe_name = args.pipe_name if args.pipe_name else "music_transfomer_pipe"

    
    music_transformer = load_model(args.path_to_model, args.compile)

    try:
        os.mkfifo(pipe_name)
        print("Creating Pipe...")
    except OSError as oe: 
        print(oe)
        if oe.errno != errno.EEXIST:
            raise
    
    try:
        while True:
            with open(pipe_name, mode='rb') as fifo:
                while True:
                    data = fifo.read()
                    if len(data) == 0:
                        print("Waiting for new data")
                        break
                    else:
                        process_midi(model_=music_transformer, inp=data, save_path=args.save_path,
                             temperature=temperature_, mode=mode_, k=k_, tempo=tempo_, verbose=args.verbose, token_count=token_count)

    except KeyboardInterrupt:
        print("Stopping...")
        os.unlink(pipe_name)
        print("Pipe removed")
        exit(0)
