//
// Created by Agata on 04.01.2022.
//


#ifndef NODES_HPP
#define NODES_HPP

#include "types.hpp"
#include "package.hpp"
#include "storage_types.hpp"
#include "helpers.hpp"
#include <memory>
#include <map>
#include <optional>
#include <utility>

enum class ReceiverType {
    RAMP, WORKER, STOREHOUSE
};

using const_iterator = std::list<Package>::const_iterator;

class IPackageReceiver {

    public:

//        IPackageReceiver() = default;

        virtual IPackageStockpile::const_iterator cbegin() const = 0;
        virtual IPackageStockpile::const_iterator cend() const = 0;
        virtual IPackageStockpile::const_iterator begin() const = 0;
        virtual IPackageStockpile::const_iterator end() const = 0;



        virtual void receive_package(Package&& p) = 0;
        virtual ElementID get_id() const = 0;
        virtual ReceiverType get_receiver_type() const = 0;

        virtual ~IPackageReceiver() = default;
};

class Storehouse: public IPackageReceiver {

    public:

        Storehouse() = default;
        Storehouse(ElementID id, std::unique_ptr<IPackageStockpile> d = std::make_unique<PackageQueue>(PackageQueue(PackageQueueType::FIFO))): id_(id), d_(std::move(d)){};

        IPackageStockpile::const_iterator cbegin() const override { return d_ -> cbegin(); }
        IPackageStockpile::const_iterator begin() const override { return d_ -> begin(); }
        IPackageStockpile::const_iterator cend() const override { return d_ -> cend(); }
        IPackageStockpile::const_iterator end() const override { return d_ -> end(); }

        void receive_package(Package&& p) override {d_->push(std::move(p));};

        ElementID get_id() const override {return id_;};
        ReceiverType get_receiver_type() const override  {return ReceiverType::STOREHOUSE;};


        IPackageStockpile* get_queue() const {return d_.get(); }

    private:
        ElementID id_;
        std::unique_ptr<IPackageStockpile> d_;


};

class ReceiverPreferences {
    public:

        using preferences_t = std::map<IPackageReceiver*, double>;
        using const_iterator = preferences_t::const_iterator;


//        ReceiverPreferences() = default;
        //ReceiverPreferences(ProbabilityGenerator pg){probability_ = pg();rand_ = std::move(pg);};

        ReceiverPreferences(ProbabilityGenerator pg = probability_generator) : rand_(std::move(pg)) {};


        const_iterator cbegin() const { return preferences_.cbegin(); }
        const_iterator cend() const { return preferences_.cend(); }
        const_iterator begin() const { return preferences_.begin(); }
        const_iterator end() const { return preferences_.end(); }


        void add_receiver(IPackageReceiver* r);
        void remove_receiver(IPackageReceiver* r);
        IPackageReceiver* choose_receiver();

        const preferences_t & get_preferences() const {return preferences_;};


    private:
//        double probability_ = 0;
        ProbabilityGenerator rand_;
        preferences_t preferences_ ;

};

class PackageSender {


    public:

        PackageSender() : receiver_preferences_(probability_generator) {};
        PackageSender(ReceiverPreferences receiver_preferences) : receiver_preferences_(std::move(receiver_preferences)){}
        PackageSender(PackageSender&&)=default;

        void send_package();
        const std::optional<Package>& get_sending_buffer() const { return (std::optional<Package>&) sending_buffer_; }


        ReceiverPreferences receiver_preferences_;

    protected:
        void push_package(Package&& p) {sending_buffer_ = std::move(p);};

    private:
        std::optional<Package> sending_buffer_ = std::nullopt;


};

class Ramp : public PackageSender {

    public:

        Ramp(ElementID id, TimeOffset di) : id_(id), di_(di) {}

        void deliver_goods(Time t);
        TimeOffset get_delivery_interval() const {return di_;}
        ElementID get_id() const {return id_; }

        ReceiverType get_receiver_type()  {return ReceiverType::RAMP;}

    private:
        ElementID id_;
        TimeOffset di_;
};

class Worker : public PackageSender, public IPackageReceiver {

public:

    Worker(ElementID id, TimeOffset pd, std::unique_ptr<IPackageQueue> q) : q_(std::move(q)), id_(id), pd_(pd), buffor_(std::nullopt) {};


    IPackageStockpile::const_iterator cbegin() const override {return q_->cbegin(); }
    IPackageStockpile::const_iterator cend() const override {return q_->cend(); }
    IPackageStockpile::const_iterator begin() const override {return q_->begin(); }
    IPackageStockpile::const_iterator end() const override {return q_->end(); }


    void do_work(Time t);
    TimeOffset get_processing_duration() const {return pd_; }
    static Time get_package_processing_start_time() {return time_; }

    ReceiverType get_receiver_type() const override { return ReceiverType::WORKER; }
    void receive_package(Package&& p) override  {q_ -> push(std::move(p)); }

    ElementID get_id() const override {return id_; }
    void prepare_package(Package&& p) { buffor_ = std::move(p); }

    IPackageQueue* get_queue() const {return q_.get();}
    std::optional<Package>& get_processing_buffer() const {return (std::optional<Package>&) buffor_; }



    static Time time_;

private:
    std::unique_ptr<IPackageQueue> q_;
    ElementID id_;
    TimeOffset pd_;
    std::optional<Package> buffor_ ;

};


#endif //NODES_HPP
