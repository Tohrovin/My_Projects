#ifndef PACKAGE_HPP
#define PACKAGE_HPP

#include <iostream>
#include <set>
#include "types.hpp"

class Package {

    public:

        Package(){

            if (!freed_IDs_.empty()) {
                ElementID first = *freed_IDs_.begin();
                freed_IDs_.erase(first);
                for (const auto el: freed_IDs_) {
                    if (el < first)
                        first = el;
                }
                id_ = first;
                assigned_IDs.insert(first);
            }
            else {
                ElementID last = *assigned_IDs.begin();
                for (const auto el: assigned_IDs) {
                    if (el > last)
                        last = el;
                }
                id_ = last + 1;
                assigned_IDs.insert(last + 1);
            }
        };


        Package(const Package&) = default;

        Package(ElementID id){
            id_ = id;
            assigned_IDs.insert(id);
            if (freed_IDs_.find(id) != freed_IDs_.end()) {
                freed_IDs_.erase(id);
            }
        };

        Package(Package && other){

            id_ = other.id_;
            other.id_ = Null_id_;
        }


        ElementID get_id() const { return id_; }

        Package& operator=(Package&& other) {

            if (id_ != Null_id_) {
            freed_IDs_.insert(id_);
            assigned_IDs.erase(id_);
    }

    id_ = other.id_;
    other.id_ = Null_id_;

    return *this;
}

        ~Package(){
            if (id_ != Null_id_) {
                if (freed_IDs_.find(id_) == freed_IDs_.end()) {
                    freed_IDs_.insert(id_);
                }
                if (assigned_IDs.find(id_) == assigned_IDs.end()) {
                    assigned_IDs.insert(id_);
                }
            }
        };


    private:

        static std::set<ElementID> assigned_IDs;
        static std::set<ElementID> freed_IDs_;

        static const ElementID Null_id_ = -1;
        ElementID id_ = Null_id_;

};

#endif //PACKAGE_HPP
