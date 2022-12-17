//
// Created by Agata on 04.01.2022.
//

#include "nodes.hpp"
#include <numeric>
#include <utility>
#include "iterator"

Time Worker::time_ = 0;

void ReceiverPreferences::add_receiver(IPackageReceiver * r) {

    preferences_.emplace(r, 1);
    for(auto & el : preferences_) {
        el.second = 1.0 / (double)(preferences_.size());
    }
}

void ReceiverPreferences::remove_receiver(IPackageReceiver * r) {

    preferences_.erase(r);
    if(preferences_.size() != 0) {

        for (auto it = preferences_.begin(); it != preferences_.end(); it++) {
            it->second = 1.0 / (double)(preferences_.size());
        }
    }
}


IPackageReceiver* ReceiverPreferences::choose_receiver() {

    IPackageReceiver *receiver = nullptr;

    double nb = rand_();
    double sum = 0;

    for(auto &it: preferences_){
        sum += it.second;
        if(nb < sum)
            return it.first;
    }

    return receiver;

}

void PackageSender::send_package() {

    if(sending_buffer_){
        IPackageReceiver* receiver = receiver_preferences_.choose_receiver();
        receiver->receive_package(std::move(sending_buffer_.value()));
        sending_buffer_.reset();
    }

}


void Ramp::deliver_goods(Time t) {

    if (std::fmod(t - 1 ,di_) == 0){

        Package Pac (id_);
        push_package(std::move(Pac));
    };
}


void Worker::do_work(Time t) {
    if(buffor_ == std::nullopt){
        if(!q_->empty()) {
            time_ = t;
            buffor_.emplace(q_->pop());
        }
        else{
            buffor_ = std::nullopt;
        }
    }
    if(buffor_ != std::nullopt) {
        if (t - time_ == pd_ - 1) {
            push_package(std::move(buffor_.value()));
            //time_ = 0;
            buffor_.reset();
            /*if (!q_->empty()) {
                time_ = t;
                prepare_package(q_->pop());
            }*/
        }
    }

}