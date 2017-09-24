#pragma once

#include <sparse-map-base/error.h>
#include <sparse-map-base/log.h>
#include "common.h"

#include <iostream>
#include <fstream>

namespace NSparseMap {

	inline void saveMatrix(const TMatrix& m, const std::string& dstFile) {
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


	inline std::vector<TMatrix> read3dArray(const std::string& srcFile) {
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

	inline void save3dArray(const std::vector<TMatrix>& m, const std::string& dstFile) {
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

} // namespace NSparseMap