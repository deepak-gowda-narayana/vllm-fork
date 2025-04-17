import argparse
import dataclasses
import os
from vllm import LLM

# from tensorizer import EncryptionParams
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    TensorizerArgs,
    tensorize_vllm_model,
)
from vllm.utils import FlexibleArgumentParser

"""
Originally taken from vLLM's `/examples/tensorize_vllm_model.py`
"""

def parse_args():
    parser = FlexibleArgumentParser(
        prog=f"python {os.path.basename(__file__)}",
        description="Script to serialize models into a optimized and (optionally) encrypted format for public model serving",
    )
    parser = EngineArgs.add_cli_args(parser)
    subparsers = parser.add_subparsers(dest='command')

    serialize_parser = subparsers.add_parser(
        'serialize', help="Serialize a model ")
    
    serialize_parser.add_argument(
        "--serialized-directory",
        type=str,
        required=False,
        default='serialized',
        help="The directory to serialize the model to. "
        "The path to where the tensors are saved is a combination of the supplied `dir` and model "
        "reference ID. For instance, if `dir` is the serialized directory, "
        "and the model HuggingFace ID is `EleutherAI/gpt-j-6B`, tensors will "
        "be saved to `dir/vllm/EleutherAI/gpt-j-6B/model.tensors`, "
        "where `suffix` is given by `--suffix` or a random UUID if not "
        "provided.")

    serialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Encrypt the model weights with a randomly-generated binary key,"
              " and save the key at this path"))
    
    deserialize_parser = subparsers.add_parser(
        'deserialize',
        help=("Deserialize a model "))
    
    deserialize_parser.add_argument(
        "--path-to-tensors",
        type=str,
        required=True,
        help="The path to the model tensors to deserialize. Like `dir/vllm/EleutherAI/gpt-j-6B/model.tensors` ")
    
    deserialize_parser.add_argument(
        "--keyfile",
        type=str,
        required=False,
        help=("Encrypt the model weights with a randomly-generated binary key,"
              " and save the key at this path"))
    
    TensorizerArgs.add_cli_args(deserialize_parser)

    return parser.parse_args()


def filter_engine_args(args: argparse.Namespace):
    # Filter out only the EngineArgs args
    eng_args_dict = {f.name: getattr(args, f.name) for f in dataclasses.fields(EngineArgs)}
    engine_args = EngineArgs.from_cli_args(argparse.Namespace(**eng_args_dict))
    return engine_args

def serialize(args: argparse.Namespace , tensorizer_config):
    engine_args = filter_engine_args(args)

    model = tensorize_vllm_model(engine_args, tensorizer_config)



def deserialize(args: argparse.Namespace , tensorizer_config):
    engine_args = filter_engine_args(args)

    print("using our code")
    llm = LLM(model=engine_args.model,
                load_format="tensorizer",
                tensor_parallel_size=args.tensor_parallel_size,
                model_loader_extra_config=tensorizer_config
    )
    return llm

if __name__ == "__main__":
    args = parse_args()

    if args.command == "serialize":
        input_dir = args.serialized_directory.rstrip('/')
        base_path = f"{input_dir}/vllm/{args.model}"
        encryption_keyfile = args.keyfile if args.keyfile else f"{base_path}/model.key"

        if args.tensor_parallel_size > 1:
            model_path = f"{base_path}/model-rank-%03d.tensors"
        else:
            model_path = f"{base_path}/model.tensors"
        
        tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=encryption_keyfile,
        )

        serialize(args , tensorizer_config)

    elif args.command == "deserialize":

        encryption_keyfile = args.keyfile if args.keyfile else None

        tensorizer_config = TensorizerConfig(
        tensorizer_uri=args.path_to_tensors,
        encryption_keyfile=encryption_keyfile,
        )
        deserialize(args , tensorizer_config)
    
    else:
        print('Serialize/Deserialize option not specified')