#include <sparse-map/model.h>
#include <sparse-map-base/log.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>

#include <yaml-cpp/yaml.h>

using namespace NSparseMap;

void saveMatrix(const TMatrix& m, const std::string& dstFile) {
	ui32 shapeSize = 2;
	ui32 rows = m.rows();
	ui32 cols = m.cols();
	
	std::ofstream outFile(dstFile, std::ios::out | std::ios::binary);
	outFile.write(reinterpret_cast<char*>(&shapeSize), sizeof(ui32));
	outFile.write(reinterpret_cast<char*>(&rows), sizeof(ui32));
	outFile.write(reinterpret_cast<char*>(&cols), sizeof(ui32));
	outFile.write(reinterpret_cast<const char*>(m.data()), cols*rows*sizeof(double));
	outFile.close();
}


std::vector<TMatrix> read3dArray(const std::string& srcFile) {
	std::ifstream inFile(srcFile, std::ios::in | std::ios::binary);

	ui32 shapeSize;
	ui32 seq;
	ui32 rows;
	ui32 cols;
	inFile.read(reinterpret_cast<char*>(&shapeSize), sizeof(ui32));

	ENSURE(shapeSize == 3, "Got unexpected shape size for 3d array: " << shapeSize);
	
	inFile.read(reinterpret_cast<char*>(&seq), sizeof(ui32));
	inFile.read(reinterpret_cast<char*>(&rows), sizeof(ui32));
	inFile.read(reinterpret_cast<char*>(&cols), sizeof(ui32));
	
	L_INFO << "Got input 3d array with shape: (" << seq << ", " << rows << ", " << cols << ")";	
	
	std::vector<TMatrix> res(seq);
	for (auto& m: res) {
		m = TMatrix(rows, cols);

		inFile.read(reinterpret_cast<char*>(m.data()), rows*cols*sizeof(double));
	}

	inFile.close();

	return res;
}

void save3dArray(const std::vector<TMatrix>& m, const std::string& dstFile) {
	ENSURE(m.size() > 0, "Saving an empty 3d array to " << dstFile);

	ui32 shapeSize = 3;
	ui32 seqSize = m.size();
	ui32 rows = m[0].rows();
	ui32 cols = m[0].cols();
	
	std::ofstream outFile(dstFile, std::ios::out | std::ios::binary);
	outFile.write(reinterpret_cast<char*>(&shapeSize), sizeof(ui32));
	outFile.write(reinterpret_cast<char*>(&seqSize), sizeof(ui32));
	outFile.write(reinterpret_cast<char*>(&rows), sizeof(ui32));
	outFile.write(reinterpret_cast<char*>(&cols), sizeof(ui32));
	for (const auto& mm: m) {
		outFile.write(reinterpret_cast<const char*>(mm.data()), cols*rows*sizeof(double));
	}
	outFile.close();
}


int main(int argc, char** argv) {
	YAML::Node config = YAML::LoadFile("/artefacts/config.yaml");

	ui32 batchSize = config["batch_size"].as<ui32>();
	ui32 inputSize = config["input_size"].as<ui32>();
	ui32 filterSize = config["filter_size"].as<ui32>();
	ui32 layerSize = config["layer_size"].as<ui32>();
	
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
		
		std::cout << windowM << "\n";
		std::cout << "\n";

		m.Tick(windowM);

		acts.push_back(m.Activation);
		mems.push_back(m.Membrane);
	}
	L_INFO << "Done";
	
	L_INFO << "Saving artefacts ...";
	save3dArray(acts, "/artefacts/activation.bin");
	save3dArray(mems, "/artefacts/membrane.bin");

	saveMatrix(m.F, "/artefacts/F.bin");
	saveMatrix(m.Fc, "/artefacts/Fc.bin");
	L_INFO << "Done";
	
	return 0;
}