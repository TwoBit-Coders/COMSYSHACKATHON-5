from model_prediction import verify_folder, verify_two_images

def main():
    print("\n Face Verification Menu")
    print("1 :- Verify using Folder")
    print("2 :- Verify two images")
    choice = input("Select an option (1/2): ").strip()

    if choice == "1":
        data_dir = input("Enter folder path for verification: ").strip()
        model_path = input("Enter path to your trained model (.pth file): ").strip()
        threshold = float(input("Enter similarity threshold (default 0.7): ") or "0.7")
        verify_folder(data_dir, model_path, threshold)

    elif choice == "2":
        img1 = input("Enter path for first image: ").strip()
        img2 = input("Enter path for second image: ").strip()
        model_path = input("Enter path to your trained model (.pth file): ").strip()
        threshold = float(input("Enter similarity threshold (default 0.7): ") or "0.7")
        verify_two_images(model_path, img1, img2, threshold)

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
