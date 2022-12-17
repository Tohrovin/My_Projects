#include <iostream>
#include "storage_types.hpp"
#include "package.hpp"

int main() {
//    std::cout<<"dupaa"<<std::endl;

    PackageQueue q(LIFO);
    q.push(Package(1));
    q.push(Package(2));
//    for (auto el: q) {
//        std::cout<<el.get_id()<<std::endl;
//    }

    Package p(std::move(q.pop()));
    std::cout<<p.get_id()<<std::endl;
    p = q.pop();
    std::cout<<p.get_id();
////    PackageQueue q(PackageQueueType::LIFO);
//    q.push(Package(1));
//    q.push(Package(2));
//
//    Package p(std::move(q.pop()));




    return 0;
}
