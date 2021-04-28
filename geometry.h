#include <boost/qvm/vec.hpp>
#include <tbb/concurrent_vector.h>

class Particle {
private:
  boost::qvm::vec<float, 3> position;
  boost::qvm::vec<float, 3> velocity;

public:
  void set_position(boost::qvm::vec<float, 3> pos);
  boost::qvm::vec<float, 3> get_position();
  void set_velocity(boost::qvm::vec<float, 3> vel);
  boost::qvm::vec<float, 3> get_velocity();
};

class Particles {
private:
  tbb::concurrent_vector<Particle> particle_list;

public:
  Particles() = default;

  Particles(float &s_x, float &s_y, float &s_z, size_t &ct);

  Particles(float *s_x, float *s_y, float *s_z, float *v_x, float *v_y,
            float *v_z, size_t &ct);

  Particles &operator=(const Particles &p) = default;

  void add_particle(Particle p);

  void add_particle(float s_x, float s_y, float s_z);

  void add_particle(float s_x, float s_y, float s_z, float v_x, float v_y,
                    float v_z);

  void add_particle(boost::qvm::vec<float, 3> loc);

  void add_particle(boost::qvm::vec<float, 3> loc,
                    boost::qvm::vec<float, 3> vel);

  Particle &get(size_t index);

  size_t get_count();

  boost::qvm::vec<float, 3> get_location(size_t index);
};

class BoundingBox {
private:
  boost::qvm::vec<float, 3> min;
  boost::qvm::vec<float, 3> max;
  boost::qvm::vec<float, 3> center;

public:
  BoundingBox(boost::qvm::vec<float, 3> min, boost::qvm::vec<float, 3> max);
  BoundingBox(float x_min, float y_min, float z_min, float x_max, float y_max,
              float z_max);
  bool contains(boost::qvm::vec<float, 3> item);

  boost::qvm::vec<float, 3> get_min();
  boost::qvm::vec<float, 3> get_max();

  void compute_center();
  boost::qvm::vec<float, 3> get_center();

};
