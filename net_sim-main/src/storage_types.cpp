#include "storage_types.hpp"

Package PackageQueue::pop() {

    if (get_queue_type() == PackageQueueType::FIFO) {
        Package package = std::move(stockpile_.front());
        stockpile_.pop_front();
        return package;

    } else {

        Package package = std::move(stockpile_.back());
        stockpile_.pop_back();
        return package;
    }
}
