#include <sparse-map/model.h>
#include <sparse-map/io_functions.h>
#include <sparse-map-base/log.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>

#include <yaml-cpp/yaml.h>

using namespace NSparseMap;


int main(int argc, char** argv) {
	YAML::Node config = YAML::LoadFile("/artefacts/config.yaml");
	
	L_INFO << "Reading config ...";
	L_INFO << config;

	ui32 batchSize = config["batch_size"].as<ui32>();
	ui32 inputSize = config["input_size"].as<ui32>();
	ui32 filterSize = config["filter_size"].as<ui32>();
	ui32 layerSize = config["layer_size"].as<ui32>();
	
	auto files = config["files"].as<std::map<std::string, std::string>>();
	auto activationStatFile = files["activation"];
	auto membraneStatFile = files["membrane"];
	auto weightsFile = files["F"];
	auto recWeightsFile = files["Fc"];

	TModel m(
		batchSize, 
		inputSize, 
		filterSize, 
		layerSize, 
		config["tau"].as<double>(),
		config["lambda"].as<double>()
	);

	std::vector<TMatrix> inputSeq = read3dArray("/artefacts/input_data.bin");
	
	ui32 seqSize = inputSeq.size();

	std::vector<TMatrix> acts;
	std::vector<TMatrix> mems;
	
	std::deque<TMatrix> window;
	for (ui32 fi=0; fi<filterSize; ++fi) {
		window.push_back(TMatrix::Zero(batchSize, inputSize));
	}
	
	TMatrix windowM = TMatrix::Zero(batchSize, filterSize*inputSize);
	L_INFO << "Running " << seqSize << " sequence";
	
	for (const auto& input: inputSeq) {
		window.pop_front();
		window.push_back(input);

		for (ui32 fi=0; fi<filterSize; ++fi) {
			windowM.block(0, fi*inputSize, batchSize, inputSize) = window[fi];
		}
		
		m.Tick(windowM);

		acts.push_back(m.Activation);
		mems.push_back(m.Membrane);
	}
	L_INFO << "Done";
	
	L_INFO << "Saving artefacts ...";
	save3dArray(acts, activationStatFile);
	save3dArray(mems, membraneStatFile);

	saveMatrix(m.F, weightsFile);
	saveMatrix(m.Fc, recWeightsFile);
	L_INFO << "Done";
	
	return 0;
}