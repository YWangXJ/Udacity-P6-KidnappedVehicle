/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;


using namespace std;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles

  //set distribution of sensor noise
  normal_distribution<double> dist_x_init(x, std[0]);
  normal_distribution<double> dist_y_init(y, std[1]);
  normal_distribution<double> dist_theta_init(theta, std[2]);
  
  //init 
  for (int i = 0; i<num_particles; i++) {
    Particle cur_particle;
    cur_particle.id = i;
    cur_particle.x = dist_x_init(gen);
    cur_particle.y = dist_y_init(gen);
    cur_particle.theta = dist_theta_init(gen);
    cur_particle.weight = 1.0;
    
    particles.push_back(cur_particle);
   }
   is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  //set distribution of sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
 
  for (int i = 0; i<num_particles; i++){
    //has to check the yaw_rate first.
    if (abs(yaw_rate) < 0.0001){
      particles[i].x += velocity * cos(particles[i].theta) * delta_t;
      particles[i].y += velocity * sin(particles[i].theta) * delta_t;
    }
    else{
      //predict
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
     
    // add noises
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations, double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
//   std::cout << "Asso p1" << std::endl;
//   std::cout << "Asso pred:"<<predicted.size()<<"Asso obser" << observations.size() << std::endl;
  for(unsigned int i = 0; i<observations.size(); i++){
    LandmarkObs obs = observations[i];
    
    //init the id for the association later
    int lm_id = -1;
//     std::cout << "Asso p2" << std::endl;
    
    //init minimum dist as large number for the comparsion later; Max of minDist can be sqrt(2)* sensor_range
    double minDist = sensor_range * sqrt(2);
    for(unsigned int j = 0; j<predicted.size(); j++){
      LandmarkObs pred = predicted[j];
      
      //calculate dist 
      double curDist = dist(obs.x,obs.y,pred.x,pred.y);
      if (curDist < minDist){
        minDist = curDist;
        lm_id = pred.id;
      }
//        std::cout << "Asso p3(i,j)"<< i<<","<<j << std::endl;
    }
    //assign the nearest id to observation
    observations[i].id = lm_id;
//    std::cout << "Asso p4" << std::endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_normalizer = 0.0;
  for (int i = 0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    //predicted landmarks within sensor range of the ith particle
    vector<LandmarkObs> predictions;
    
    //land mark
    for (unsigned int j = 0; j<map_landmarks.landmark_list.size(); j++) {
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      
      if(dist(lm_x,lm_y,p_x,p_y) <= sensor_range){        
        //add to prediction vector
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }
//     std::cout << "p1"<< std::endl;
    
    //transformation of the observations using (lesson5.15)
    vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j<observations.size(); j++){
      double m_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double m_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, m_x, m_y });
    }
//     std::cout << "p2"<< std::endl;
    // dataAssociation between predictions and transformed observations of ith particle
    dataAssociation(predictions, transformed_obs, sensor_range);
//     std::cout << "p3"<< std::endl;
    //Reset the weight of particle to 1.0 for each step
    particles[i].weight = 1.0;
    
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      // init jth observation and pridiction
      double obs_x, obs_y, pred_x, pred_y;
      obs_x = transformed_obs[j].x;
      obs_y = transformed_obs[j].y;
      
      int associated_pred = transformed_obs[j].id;
      //search for the associated predictions based on jth observation
       for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_pred) {
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
          
          //calculate weight
          double std_x = std_landmark[0];
          double std_y = std_landmark[1];
          double obs_weight = (1/(2*M_PI*std_x*std_y)) * exp( -( pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2)))));
      
    	    //continiously multiply the current weight of current particle 
          particles[i].weight *= obs_weight;
        }
//          std::cout << "p4"<< std::endl;
       }
//       std::cout << "p5"<< std::endl;
    }
    weight_normalizer += particles[i].weight;
//     std::cout << "p6"<< std::endl;
  }
    //normalize the weights
    for (unsigned int i = 0; i < particles.size(); i++) {
      particles[i].weight /= weight_normalizer;
      //weights[i] = particles[i].weight;
    }
//   std::cout << "p7"<< std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  //create list of resamped particles to be filled
  vector<Particle> resampled_p;
  
  // get all normalized weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  //ramdom begining index of the resampling wheel
  uniform_int_distribution<int> intdist(0, num_particles - 1);//index starts from 0 to num_particles-1
  
  int index = intdist(gen);
  
  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
   
  //init the turning angle
  double beta = 0.0;
    
  // start resampling
  for (int i = 0; i < num_particles; i++) {
    //ramdom on double 
    uniform_real_distribution<double> realdist(0.0, max_weight);
    beta += realdist(gen) * 2.0;
    
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;//mod over num_particles, cicle the index
    }
    resampled_p.push_back(particles[index]);
  }

  particles = resampled_p;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}