#include "data.h"

#include <boost/qvm/vec_access.hpp>

using namespace boost::qvm;

DataOutput::DataOutput(std::string &filename) {
    output_stream.open(filename.c_str());
}

DataOutput::~DataOutput() {
    output_stream.close();
}

void DataOutput::write_bounding_box(BoundingBox *bb) {
    boost::qvm::vec<float, 3> min = bb->get_min();
    boost::qvm::vec<float, 3> max = bb->get_max();
    output_stream << "BOUNDINGBOX " << A<0>(min) << " " << A<1>(min) << " " << A<2>(min) << " "
        << A<0>(max) << " " << A<1>(max) << " " << A<2>(max) << std::endl;
}

void DataOutput::write_datapoint(boost::qvm::vec<float, 3> pos) {
    output_stream << "DATAPOINT " << A<0>(pos) << " " << A<1>(pos) << " " << A<2>(pos) << std::endl;
}


void DataOutput::add_datapoints(Particles &particles) {
    for(int i = 0; i < particles.get_count(); i++)
        write_datapoint(particles.get(i).get_position());
}

void DataOutput::add_octree_bounding_boxes(OctreeNode *parent) {
    // Write the current bounding box
    write_bounding_box(&(parent->get_bounding_box()));

    // Recurse down through children
    for(int i = 0; i < 8; i++) {
        if (parent->get_particle_count() > 1) {
            OctreeNode *child = parent->get_child(i);
            if(child->get_particle_count() > 0)
                add_octree_bounding_boxes(child);
        }
    }
}