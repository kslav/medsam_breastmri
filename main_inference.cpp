//Set up the environment

#include <iostream>
#include <torch/script.h>
#include <memory>
//#include <opencv2/opencv.hpp>

using namespace std;

//Purpose: To load a trained model, load input images, and run inference to obtain a prediction 
//Status: under development, configruing libtorch library and resolving compiler issues
int main()
{
	// First, test that the environment is built correctly by simply printing a torch tensor. 
	// Comment this section out once you've verified it works!
	cout << "Testing that the environment loads\n";
  	torch::Tensor tensor = torch::rand({2, 3});
  	cout << tensor << endl;

  	// New-ish to C++, working on writing the rest
}
