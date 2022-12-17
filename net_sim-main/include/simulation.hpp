//
// Created by Agata on 17.01.2022.
//

#ifndef NET_SIM_SIMULATION_HPP
#define NET_SIM_SIMULATION_HPP

#include "factory.hpp"

void simulate(Factory& f,TimeOffset d,const std::function<void (Factory&, Time)>& rf);


#endif //NET_SIM_SIMULATION_HPP
