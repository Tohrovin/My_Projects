//
// Created by Agata on 04.01.2022.
//

#ifndef FACTORY_HPP
#define FACTORY_HPP

#include "types.hpp"
#include "nodes.hpp"
#include "package.hpp"
#include "storage_types.hpp"
#include <algorithm>
#include <optional>
#include "string"
#include <list>
#include <sstream>

template <class Node>
class NodeCollection {

    public:
        using container_t = typename std::list<Node>;
        using iterator = typename container_t::iterator;
        using const_iterator = typename container_t::const_iterator;


        iterator begin() {return nlist_.begin(); }
        iterator end() {return nlist_.end(); }
        const_iterator cbegin() const {return nlist_.cbegin(); }
        const_iterator cend() const {return nlist_.cend(); }
        const_iterator begin() const {return nlist_.begin(); }
        const_iterator end() const {return nlist_.end(); }

        void add(Node& node) {nlist_.push_back(std::move(node)); }

        void remove_by_id(ElementID id){

            auto it = find_by_id(id);
            if (it != nlist_.end()) {
                nlist_.erase(it);
            }
        };

        iterator find_by_id(ElementID id_) {

            auto it = std::find_if(nlist_.begin(), nlist_.end(),
                                   [id_](const auto& elem) { return (elem.get_id() == id_); });
            return it;
        };

        const_iterator find_by_id(ElementID id_) const {

            auto it = std::find_if(nlist_.cbegin(), nlist_.cend(), [id_](const auto& elem){ return (elem.get_id() == id_);});
            return it;
        };

    private:

        container_t nlist_;

};


class Factory {

    public:

        void add_ramp(Ramp&& r) { ramps_list_.add(r); }

        void remove_ramp(ElementID id){ramps_list_.remove_by_id(id);};

        NodeCollection<Ramp>::iterator find_ramp_by_id(ElementID id) {return ramps_list_.find_by_id(id); }
        NodeCollection<Ramp>::const_iterator find_ramp_by_id(ElementID id) const {return ramps_list_.find_by_id(id); }


        NodeCollection<Ramp>::const_iterator ramp_cbegin() const {return ramps_list_.cbegin(); }
        NodeCollection<Ramp>::const_iterator ramp_cend() const {return ramps_list_.cend(); }
        NodeCollection<Ramp>::const_iterator ramp_begin() const {return ramps_list_.begin(); }
        NodeCollection<Ramp>::const_iterator ramp_end() const {return ramps_list_.end(); }
        NodeCollection<Ramp>::iterator ramp_begin()  {return ramps_list_.begin(); }
        NodeCollection<Ramp>::iterator ramp_end()  {return ramps_list_.end(); }


        void add_storehouse(Storehouse&& s) {storeh_list_.add(s); }

        void remove_storehouse(ElementID id);
        NodeCollection<Storehouse>::iterator find_storehouse_by_id(ElementID id) {return storeh_list_.find_by_id(id); }
        NodeCollection<Storehouse>::const_iterator find_storehouse_by_id(ElementID id) const {return storeh_list_.find_by_id(id); }


        NodeCollection<Storehouse>::const_iterator storehouse_cbegin() const {return storeh_list_.cbegin(); }
        NodeCollection<Storehouse>::const_iterator storehouse_cend() const {return storeh_list_.cend(); }
        NodeCollection<Storehouse>::const_iterator storehouse_begin() const {return storeh_list_.begin(); }
        NodeCollection<Storehouse>::const_iterator storehouse_end() const {return storeh_list_.end(); }
        NodeCollection<Storehouse>::iterator storehouse_begin()  {return storeh_list_.begin(); }
        NodeCollection<Storehouse>::iterator storehouse_end()  {return storeh_list_.end(); }


        void add_worker(Worker&& w) {workers_list_.add(w);};
        void remove_worker(ElementID id);

        NodeCollection<Worker>::iterator find_worker_by_id(ElementID id) {return workers_list_.find_by_id(id);};
        NodeCollection<Worker>::const_iterator find_worker_by_id(ElementID id) const {return workers_list_.find_by_id(id);};

        NodeCollection<Worker>::const_iterator worker_cbegin() const {return workers_list_.cbegin(); }
        NodeCollection<Worker>::const_iterator worker_cend() const {return workers_list_.cend(); }
        NodeCollection<Worker>::const_iterator worker_begin() const {return workers_list_.begin(); }
        NodeCollection<Worker>::const_iterator worker_end() const {return workers_list_.end(); }
        NodeCollection<Worker>::iterator worker_begin() {return workers_list_.begin(); }
        NodeCollection<Worker>::iterator worker_end() {return workers_list_.end(); }


        bool is_consistent() const;
        void do_deliveries(Time t);
        void do_package_passing();
        void do_work(Time t);

    private:
        NodeCollection<Worker> workers_list_;
        NodeCollection<Ramp> ramps_list_;
        NodeCollection<Storehouse> storeh_list_;
        template <class Node>

        void remove_receiver(NodeCollection<Node>& collection, ElementID id);
};




class ExceptionForHasReachable : public std::logic_error {
public:
    ExceptionForHasReachable() : std::logic_error("nope") {}
};

enum class ElementType{
    LOADING_RAMP, WORKER, STOREHOUSE, LINK
};

struct ParsedLineData {
    ElementType element_type;
    std::map<std::string, std::string> parameters;
};

ParsedLineData parse_line(const std::string& line);
Factory load_factory_structure(std::istream& is);
void save_factory_structure(Factory& factory, std::ostream& os);

#endif //FACTORY_HPP