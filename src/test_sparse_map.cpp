#include <sparse-map/model.h>
#include <sparse-map-base/log.h>

#include <iostream>
#include <fstream>
#include <vector>

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


int main(int argc, char** argv) {
	YAML::Node config = YAML::LoadFile("/artefacts/config.yaml");

	ui32 batchSize = config["batch_size"].as<ui32>();
	ui32 inputSize = config["input_size"].as<ui32>();
	ui32 filterSize = config["filter_size"].as<ui32>();
	ui32 layerSize = config["layer_size"].as<ui32>();
	
	TModel m(batchSize, inputSize, filterSize, layerSize);

	std::std::vector<TMatrix> inputSeq = read3dArray("/artefacts/input_data.bin");
	
	ui32 seqSize = inputSeq.size();

	std::vector<TMatrix> acts;
	
	std::vector<TMatrix> window(filter_size);

	for (ui32 idx=0; idx < seq_size; ++idx) {
		m.Tick(x);
	}

	return 0;
}