import argparse
from pathlib import Path

from MT_Inter import main as MT_Inter_main
from MT_Cog import main as MT_Cog_main
from MT_App import main as MT_App_main


def main():
    parser = argparse.ArgumentParser(description="execute three types of QA tests")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="input data directory path (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="output directory path (default: ./output)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model_name",
        help="test model name, will be used as output subfolder (default: deepseek-r1-huawei-910b)",
    )
    
    # === model and API configuration ===
    parser.add_argument(
        "--attack-api-key",
        type=str,
        default="api_key",
        help="attack model API Key",
    )
    parser.add_argument(
        "--attack-base-url",
        type=str,
        default="base_url",
        help="attack model Base URL",
    )
    parser.add_argument(
        "--attack-model-name",
        type=str,
        default="attack_model_name",
        help="attack model name (default: deepseek-r1-huawei-910b)",
    )
    parser.add_argument(
        "--defense-api-key",
        type=str,
        default="api_key",
        help="defense model API Key",
    )
    parser.add_argument(
        "--defense-base-url",
        type=str,
        default="base_url",
        help="defense model Base URL",
    )
    parser.add_argument(
        "--defense-model-name",
        type=str,
        default="defense_model_name",
        help="defense model name (default: deepseek-v3-huawei-910b)",
    )
    
    args = parser.parse_args()
    
    # build input file paths
    data_dir = Path(args.data_dir)
    MT_Inter_path = data_dir / "MT_Inter.xlsx"
    MT_Cog_path = data_dir / "MT_Cog.xlsx"
    MT_App_path = data_dir / "MT_App.xlsx"
    
    print("=" * 60)
    print("start executing three types of QA tests")
    print(f"data directory: {data_dir}")
    print(f"output directory: {args.output_dir}")
    print(f"model name: {args.model_name}")
    print(f"attack model: {args.attack_model_name} ({args.attack_base_url})")
    print(f"defense model: {args.defense_model_name} ({args.defense_base_url})")
    print("=" * 60)
    
    # execute three tests sequentially
    print("\n[1/3] start executing: MT_Inter test")
    print("-" * 60)
    try:
        MT_Inter_main(
            input_data_path=str(MT_Inter_path),
            output_dir=args.output_dir,
            model_name=args.model_name,
            attack_api_key=args.attack_api_key,
            attack_base_url=args.attack_base_url,
            attack_model_name=args.attack_model_name,
            defense_api_key=args.defense_api_key,
            defense_base_url=args.defense_base_url,
            defense_model_name=args.defense_model_name,
        )
        print(f"✓ MT_Inter test completed")
    except Exception as e:
        print(f"✗ MT_Inter test failed: {e}")
    
    print("\n[2/3] start executing: MT_Cog test")
    print("-" * 60)
    try:
        MT_Cog_main(
            input_data_path=str(MT_Cog_path),
            output_dir=args.output_dir,
            model_name=args.model_name,
            attack_api_key=args.attack_api_key,
            attack_base_url=args.attack_base_url,
            attack_model_name=args.attack_model_name,
            defense_api_key=args.defense_api_key,
            defense_base_url=args.defense_base_url,
            defense_model_name=args.defense_model_name,
        )
        print(f"✓ MT_Cog test completed")
    except Exception as e:
        print(f"✗ MT_Cog test failed: {e}")
    
    print("\n[3/3] start executing: MT_App test")
    print("-" * 60)
    try:
        MT_App_main(
            input_data_path=str(MT_App_path),
            output_dir=args.output_dir,
            model_name=args.model_name,
            attack_api_key=args.attack_api_key,
            attack_base_url=args.attack_base_url,
            attack_model_name=args.attack_model_name,
            defense_api_key=args.defense_api_key,
            defense_base_url=args.defense_base_url,
            defense_model_name=args.defense_model_name,
        )
        print(f"✓ MT_App test completed")
    except Exception as e:
        print(f"✗ MT_App test failed: {e}")
    
    print("\n" + "=" * 60)
    print("all tests executed successfully!")
    print(f"output file located at: {Path(args.output_dir) / args.model_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

