#include <iostream>
#include <fstream>
#include <boost/qvm/vec.hpp>

#include "generator.h"

class DataOutput {
private:
    std::ofstream file_stream;
public:
    DataOutput(char* filename);
    ~DataOutput();
    void write_bounding_box(boost::qvm::vec<float, 3> start, boost::qvm::vec<float, 3> end);
    void write_datapoint(boost::qvm::vec<float, 3> pos);
    void add_datapoints(Particles *particles);
    void add_octree_bounding_boxes(OctreeNode *parent);
};
