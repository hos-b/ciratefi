#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

int main(int, char **)
{
    Eigen::Tensor<double, 3> tensor(3, 5, 5);
    // clang-format: off
    tensor.setValues({{{10, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0},
                       {0, 0, 0, 0, 0}},
                      {{10, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1},
                       {1, 1, 1, 1, 1}},
                      {{7, 2, 2, 2, 2},
                       {2, 2, 2, 2, 2},
                       {2, 2, 2, 2, 2},
                       {2, 2, 2, 2, 2},
                       {2, 2, 2, 2, 2}}});
    // clang-format: on
    std::cout << tensor << std::endl;
    Eigen::Tensor<double, 2> mean = tensor.mean(Eigen::array<int, 1>({0}));
    std::cout << "mean =\n" << mean << std::endl;
    Eigen::Tensor<long int, 2> argmax = tensor.argmax(0);
    // std::cout << "argmax =\n" << argmax << std::endl;
    return 0;
}