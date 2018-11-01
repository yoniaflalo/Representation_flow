from Representation_flow import *
import time


def main():
    num_classes = 100
    RF = Representation_flow(num_classes)
    test_tensor = torch.rand([16, 4, 3, 256, 256])
    if torch.cuda.is_available():
        RF = RF.cuda()
        test_tensor = test_tensor.cuda()
    print("Starting Representation flow")
    start = time.time()
    f = RF(test_tensor)
    end = time.time()
    print(f"Elapsed time : {(end - start)} seconds")


if __name__ == "__main__":
    main()
