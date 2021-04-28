#include "data.h"

#include <boost/qvm/vec_access.hpp>

using namespace boost::qvm;

DataOutput::DataOutput(char* filename) {
    output_stream.open(filename);
}

DataOutput::~DataOutput() {
    output_stream.close();
}

void write_bounding_box(BoundingBox *bb) {
    boost::qvm::vec<float, 3> min = bb->get_min();
    boost::qvm::vec<float, 3> end = bb->get_max();
    output_stream << "BOUNDINGBOX " << A<0>(start) << " " << A<1>(start) << " " << A<2>(start) << " "
        A<0>(end) << " " << A<1>(end) << " " << A<2>(end) << std::endl:
}

void write_datapoint(boost::qvm::vec<float, 3> pos) {
    output_stream << "DATAPOINT " << A<0>(pos) << " " << A<1>(pos) << " " << A<2>(pos) << std::endl;
}


void add_datapoints(Particles *particles) {
    for(int i = 0; i < particles->get_count(); i++)
        write_datapoint(particles->get(i)->get_position());
}

void add_octree_bounding_boxes(OctreeNode *parent) {
    // Write the current bounding box
    write_bounding_box(parent->get_bounding_box());

    // Recurse down through children
    for(int i = 0; i < 8; i++) {
        OctreeNode *child = parent->get_child(i);
        if(child->get_particle_count() > 1)
            add_octree_bounding_boxes(child);
    }
}