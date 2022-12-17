//
// Created by Agata on 04.01.2022.
//

#include "factory.hpp"

enum class NodeColor {
    UNVISITED,
    VISITED,
    VERIFIED };



void Factory::remove_worker(ElementID id) {
    remove_receiver(ramps_list_, id);
    remove_receiver(workers_list_, id);
    workers_list_.remove_by_id(id);

}


void Factory::remove_storehouse(ElementID id) {
    remove_receiver(workers_list_, id);
    storeh_list_.remove_by_id(id);
}


bool has_reachable_storehouse(PackageSender* sender, std::map<const PackageSender*, NodeColor>& node_colors){

    if(node_colors[sender] == NodeColor::VERIFIED)
        return true;

    node_colors[sender] = NodeColor::VISITED;

    if(sender->receiver_preferences_.get_preferences().empty()) {
        throw ExceptionForHasReachable();
    }

    bool has_other_receiver = false;

    for(auto& it: sender->receiver_preferences_.get_preferences()){


        if(it.first->get_receiver_type() == ReceiverType::STOREHOUSE){
            has_other_receiver = true;
        }
        else if(it.first->get_receiver_type() == ReceiverType::WORKER){

            IPackageReceiver* receiver_ptr = it.first;
            auto worker_ptr = dynamic_cast<Worker*>(receiver_ptr);
            auto sd_ptr = dynamic_cast<PackageSender*>(worker_ptr);
            if(sd_ptr == sender)
                continue;

            has_other_receiver = true;

            if(node_colors[sd_ptr] == NodeColor::UNVISITED)
                has_reachable_storehouse(sd_ptr, node_colors);
        }
    }

    node_colors[sender] = NodeColor::VERIFIED;

    if(has_other_receiver){
        return true;
    }
    else{
        throw ExceptionForHasReachable();
    }
}




bool Factory::is_consistent() const {


    std::map<const PackageSender*, NodeColor> colors;

    for(auto& it: workers_list_)
        colors[(PackageSender *) &it] = NodeColor::UNVISITED;
    for(auto& it: ramps_list_)
        colors[(PackageSender* ) &it] = NodeColor::UNVISITED;
    for(auto& it: ramps_list_)
    {
        try {
            has_reachable_storehouse((PackageSender *) &it, colors);
            return true;
        }
        catch(ExceptionForHasReachable& exception){
            return false;
        }
    }
    return true;
}


void Factory::do_deliveries(Time t) {

    for (auto &ramp : ramps_list_){
        ramp.deliver_goods(t);
    }
}



void Factory::do_package_passing() {

    for (auto &ramp : ramps_list_){
        ramp.send_package();
    }

    for (auto &worker : workers_list_){
        worker.send_package();
    }
}

void Factory::do_work(Time t) {

    for (auto &worker : workers_list_){
        worker.do_work(t);
    }
}

template<class Node>
void Factory::remove_receiver(NodeCollection<Node>& collection, ElementID id) {

    for (auto &elem: collection) {

        for(auto& it: elem.receiver_preferences_.get_preferences()) {

            if (it.first->get_id() == id) {
                elem.receiver_preferences_.remove_receiver(it.first);
                break;
            }
        }
    }
}






ParsedLineData parse_line(const std::string& line) {

    std::vector<std::string> many_toks;
    std::string token;
    std::istringstream token_stream(line);
    char delimiter = ' ';

    while (std::getline(token_stream, token, delimiter)) {
        many_toks.push_back(token);
    }


    std::string type = many_toks.front();
    many_toks.erase(many_toks.begin());
    ParsedLineData parse_data = ParsedLineData();


    if(type == "LOADING_RAMP") {
        parse_data.element_type = ElementType::LOADING_RAMP;
    }
    else if(type == "WORKER") {
        parse_data.element_type = ElementType::WORKER;
    }
    else if(type == "LINK") {
        parse_data.element_type = ElementType::LINK;
    }
    else if(type == "STOREHOUSE") {
        parse_data.element_type = ElementType::STOREHOUSE;
    }

    for(const auto& el: many_toks) {

        std::vector<std::string> key_value;
        std::string tok;
        std::istringstream single_token(el);
        char del = '=';

        while (std::getline(single_token, tok, del)) {
            key_value.push_back(tok);
        }

        parse_data.parameters[key_value[0]] = key_value[1];
    }
    return parse_data;
}


Factory load_factory_structure(std::istream& is) {

    Factory factory;
    std::string line;

    while (std::getline(is, line)) {
        if(line[0] == ';' or line.empty()) {
            continue;
        }
        ParsedLineData parsed_line = parse_line(line);

        if(parsed_line.element_type == ElementType::LOADING_RAMP) {
            ElementID id = 0;
            TimeOffset di = 0;
            for(const auto&el: parsed_line.parameters){

                if(el.first == "id") {
                    id = std::stoi(el.second);
                }
                else if(el.first == "delivery-interval") {
                    di = std::stoi(el.second);
                }
            }
            factory.add_ramp(Ramp(id, di));
        }


        else if(parsed_line.element_type == ElementType::WORKER) {

            PackageQueueType q;
            ElementID id = 0;
            TimeOffset pd = 0;

            for (const auto &el: parsed_line.parameters) {
                if (el.first == "id"){
                    id = std::stoi(el.second);
                }
                else if (el.first == "processing-time") {
                    pd = std::stoi(el.second);
                }
                else if (el.first == "queue-type") {
                    if (el.second == "FIFO") {
                        q = PackageQueueType::FIFO;
                    }
                    else if (el.second == "LIFO"){
                        q = PackageQueueType::LIFO;
                    }
                }
            }

            std::unique_ptr<IPackageQueue> qq = std::make_unique<PackageQueue>(q);
            factory.add_worker(Worker(id, pd, std::move(qq)));


        }
        else if(parsed_line.element_type == ElementType::STOREHOUSE) {
            factory.add_storehouse(Storehouse(std::stoi(parsed_line.parameters["id"])));
        }

        else if(parsed_line.element_type == ElementType::LINK){

            std::string src;
            std::string src_id;
            std::string dest;
            std::string dest_id;

            for(const auto& el: parsed_line.parameters) {

                if (el.first == "src") {
                    std::vector<std::string> key_value;
                    std::string token;
                    std::istringstream single_token(el.second);
                    char del = '-';

                    while (std::getline(single_token, token, del)) {
                        key_value.push_back(token);
                    }

                    src = key_value[0];
                    src_id = key_value[1];
                }

                else if (el.first == "dest") {

                    std::vector<std::string> key_value;
                    std::string token;
                    std::istringstream single_token(el.second);
                    char del = '-';

                    while (std::getline(single_token, token, del)) {
                        key_value.push_back(token);
                    }

                    dest = key_value[0];
                    dest_id = key_value[1];
                }
            }


            if(src == "worker"){

                if(dest == "store") {

                    auto worker = factory.find_worker_by_id(std::stoi(src_id));
                    auto storehouse = factory.find_storehouse_by_id(std::stoi(dest_id));

                    worker->receiver_preferences_.add_receiver(&(*storehouse));
                }

                else if(dest == "worker"){

                    auto worker = factory.find_worker_by_id(std::stoi(src_id));
                    auto worker2 = factory.find_worker_by_id(std::stoi(dest_id));

                    worker->receiver_preferences_.add_receiver(&(*worker2));
                }
            }

            else if(src == "ramp"){

                if(dest == "worker") {

                    auto ramp = factory.find_ramp_by_id(std::stoi(src_id));
                    auto worker = factory.find_worker_by_id(std::stoi(dest_id));

                    ramp->receiver_preferences_.add_receiver(&(*worker));
                }
                else if(dest == "store") {

                    auto ramp = factory.find_ramp_by_id(std::stoi(src_id));
                    auto storehouse = factory.find_storehouse_by_id(std::stoi(dest_id));

                    ramp->receiver_preferences_.add_receiver(&(*storehouse));
                }
            }
        }
    }

    return factory;
}


void save_factory_structure(Factory& factory, std::ostream& os) {

    if (factory.ramp_begin() != factory.ramp_end()) {
        os << "; == LOADING_RAMPS ==" << '\n' << '\n';
        for (auto it = factory.ramp_begin(); it != factory.ramp_end(); it++) {

            os << "LOADING_RAMP " << "id=" << std::to_string(it->get_id()) << " delivery-interval="
               << std::to_string(it->get_delivery_interval());
            os << '\n';
        }
    }

    os << '\n';
    if (factory.worker_begin() != factory.worker_end()) {

        os << "; == WORKERS ==" << '\n' << '\n';

        for (auto it = factory.worker_begin(); it != factory.worker_end(); it++) {

            os << "WORKER " << "id=" << std::to_string(it->get_id()) << " processing-time="

               << std::to_string(it->get_processing_duration()) << " queue-type=";

            if (it->get_queue()->get_queue_type() == PackageQueueType::FIFO)
                os << "FIFO";
            else if (it->get_queue()->get_queue_type() == PackageQueueType::LIFO)
                os << "LIFO";
            os << '\n';
        }
    }

    os << '\n';

    if (factory.storehouse_begin() != factory.storehouse_end()) {
        os << "; == STOREHOUSES ==" << '\n' << '\n';

        for (auto it = factory.storehouse_begin(); it != factory.storehouse_end(); it++) {
            os << "STOREHOUSE " << "id=" << std::to_string(it->get_id());
            os << '\n';

        }
    }

    os << '\n';
    os << "; == LINKS ==" << '\n';
    os << '\n';

    if (factory.ramp_begin() != factory.ramp_end()) {

        for (auto it = factory.ramp_begin(); it != factory.ramp_end(); it++) {
            for(auto el = it->receiver_preferences_.get_preferences().rbegin(); el != it->receiver_preferences_.get_preferences().rend(); el++){

                os << "LINK " << "src=ramp-" << it->get_id() << " dest=";
                if(el->first->get_receiver_type() == ReceiverType::WORKER)
                    os << "worker-" << el->first->get_id();

                else if(el->first->get_receiver_type() == ReceiverType::STOREHOUSE)
                    os << "store-" << el->first->get_id();
                os << "\n";
            }
            os << "\n";
        }
    }

    if (factory.worker_begin() != factory.worker_end()) {

        for (auto it = factory.worker_begin(); it != factory.worker_end(); it++) {

            for(auto el = it->receiver_preferences_.get_preferences().rbegin(); el != it->receiver_preferences_.get_preferences().rend(); el++){
                os << "LINK " << "src=worker-" << it->get_id() << " dest=";

                if(el->first->get_receiver_type() == ReceiverType::WORKER)
                    os << "worker-" << el->first->get_id();
                else if(el->first->get_receiver_type() == ReceiverType::STOREHOUSE)
                    os << "store-" << el->first->get_id();
                os << "\n";
            }

            os << '\n';
        }
    }
}