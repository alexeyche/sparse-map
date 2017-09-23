#include <sparse_map/model.h>

#include <iostream>

#include <yaml-cpp/yaml.h>

using namespace NSparseMap;

int main(int argc, char** argv) {
	YAML::Node config = YAML::LoadFile("config.yaml");

	ui32 batchSize = config["batch_size"].as<ui32>();
	// ui32 inputSize = config["input_size"].as<ui32>();
	ui32 layerSize = config["layer_size"].as<ui32>();
	
	Model m(batchSize, layerSize);

	return 0;
}