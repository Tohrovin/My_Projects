
#ifndef STORAGE_TYPES_HPP
#define STORAGE_TYPES_HPP


#include "package.hpp"
#include <list>
#include <deque>
#include <optional>


class IPackageStockpile {
public:
    IPackageStockpile() = default;

    using const_iterator = std::list<Package>::const_iterator;

//    const_iterator cbegin() const { return Stockpile_.cbegin(); };
//    const_iterator cend() const { return Stockpile_.cend(); };
//    const_iterator begin() const { return Stockpile_.begin(); };
//    const_iterator end() const { return Stockpile_.end(); };

    virtual const_iterator cbegin() const = 0;
    virtual const_iterator cend() const = 0;
    virtual const_iterator begin() const = 0;
    virtual const_iterator end() const = 0;

    virtual void push(Package&& Pac) { Stockpile_.emplace_back(Pac); };
    virtual bool empty() const { return Stockpile_.empty(); };
    virtual size_t size() const { return Stockpile_.size(); };


    std::list<Package> Stockpile_ = {};

    virtual ~IPackageStockpile() = default;

};

enum class PackageQueueType {
    FIFO, LIFO
};




class IPackageQueue : public IPackageStockpile {
public:

    IPackageQueue() = default;

    virtual Package pop() = 0;
    virtual PackageQueueType get_queue_type() const = 0;

    ~IPackageQueue() override = default;
};



class PackageQueue : public IPackageQueue {

public:

    PackageQueue() = default;
    PackageQueue(PackageQueueType type): type_(type), stockpile_(0){};
    void push(Package&& package) override { stockpile_.emplace_back(std::move(package)); }

    bool empty() const override { return stockpile_.empty(); }
    Package pop() override;

    PackageQueueType get_queue_type() const override {return type_;};
    std::list<Package>::size_type size() const override { return stockpile_.size(); }



    const_iterator cbegin() const override { return stockpile_.cbegin(); }
    const_iterator begin() const override { return stockpile_.begin(); }
    const_iterator cend() const override { return stockpile_.cend(); }
    const_iterator end() const override { return stockpile_.end(); }

private:

    PackageQueueType type_;
    std::list<Package> stockpile_;

};

#endif //STORAGE_TYPES_HPP