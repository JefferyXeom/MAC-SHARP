//// For file operations
#include <fstream> // For file stream operations
#include <iomanip> // For input/output manipulators, including std::setprecision
#include <sys/stat.h> // For file status, including permissions (System dependent)
#include <yaml-cpp/yaml.h> // For YAML file operations, including reading and writing
//// For input/output operations, POSIX systems arguments and system call wrappers
#include <iostream> // For input/output operations
#include <getopt.h> // For command-line option parsing (System dependent)
#include <unistd.h> // For POSIX system calls (System dependent)
//// For string operations
#include <string> // For string operations
//// For exit function
#include <cstdlib> // For exit function

////////
// Gloval variable declarations
// TODO: Check if these global variables are necessary or can be encapsulated in main function
////////

std::string folder_path = "./dataset"; // Default folder path for dataset
bool verbose = false; // Verbose mode flag
bool overwrite = false; // Overwrite existing output file flag
bool add_overlap = false; // Add overlap flag (pending)
bool low_inlier_ratio = false; // Low inlier flag (pending)
bool no_logs = false; // No logs flag

std::string called_program_name = "MAC_SHARP"; // Name of the called program in system calls(?)

// Evaluation metrics
double RE, TE, success_estimate_rate;
std::vector<int> scene_num;
int cnt;



////////
// Function declarations
////////

// Dataset configuration loading function
// TODO: Set up general dataset configuration loading rather than only threeDMatch, threeDlomatch, and ETH datasets 20250614
void loadDatasetConfig() {
    YAML::Node config = YAML::LoadFile("config.yaml");
    auto threeDMatch = config["datasets"]["threeDMatch"];
    auto threeDlomatch = config["datasets"]["threeDlomatch"];
    auto ETH = config["datasets"]["ETH"];
}


void showDatasetConfig(const std::string& dataset_name) {
    // Load the dataset configuration from the YAML file
    YAML::Node config = YAML::LoadFile("config.yaml");
    
    // Check if the dataset exists in the configuration
    if (config["datasets"][dataset_name]) {
        auto dataset = config["datasets"][dataset_name];
        std::cout << "Dataset: " << dataset_name << std::endl;
        std::cout << "Description: " << dataset["description"].as<std::string>() << std::endl;
        std::cout << "Number of scenes: " << dataset["num_scenes"].as<int>() << std::endl;
        // Add more fields as necessary
    } else {
        std::cerr << "Dataset '" << dataset_name << "' not found in configuration." << std::endl;
    }
}
std::vector<std::string> analyse(const std::string& name){



    return std::vector<std::string>();
}


void demo(){
    std::cout << "This is a demo function." << std::endl;
    // Add demo functionality here
    std::cout << "Demo completed." << std::endl;
}

void usage(){
    std::cout << "Usage:" << std::endl;
    std::cout << "\tHELP --help" <<std::endl;
    std::cout << "\tDEMO --demo" << std::endl;
    std::cout << "\tREQUIRED ARGS:" << std::endl;
    std::cout << "\t\t--output_path\toutput path for saving results." << std::endl;
    std::cout << "\t\t--input_path\tinput data path." << std::endl;
    std::cout << "\t\t--dataset_name\tdataset name. [3dmatch/3dlomatch/KITTI]" << std::endl;
    std::cout << "\t\t--descriptor\tdescriptor name. [fpfh/fcgf/predator]" << std::endl;
    std::cout << "\t\t--start_index\tstart from given index. (begin from 0)" << std::endl;
    std::cout << "\tOPTIONAL ARGS:" << std::endl;
    std::cout << "\t\t--no_logs\tforbid generation of log files." << std::endl;
};


int main(int argc, char** argv) {

    ////////
    // Initialize variables
    ////////
    add_overlap = false;
    low_inlier_ratio = false;
    no_logs = false;
    int id = 0;
    std::string output_path; //程序生成文件的保存目录
    std::string dataset_path; //数据集路径
    std::string dataset_name; //数据集名称


    ////////
    std::vector<double> scene_re_num;
    std::vector<double> scene_te_num;
    std::vector<double> scene_success_rate;
    int correct_num = 0;
    int total_num = 0;
    double total_re = 0.0;
    double total_te = 0.0;

    ////////
    // Parse command-line arguments
    // Using getopt_long for long options parsing
    // Note: getopt_long is a POSIX function, so this code is system dependent
    // If you are using a different system, you may need to adjust the includes and function calls accordingly
    ////////
    int opt;
    int digit_opind = 0;
    int option_index = 0;
    static struct option long_options[] = {
            {"output_path", required_argument, NULL, 'o'},
            {"input_path", required_argument, NULL, 'i'},
            {"dataset_config", required_argument, NULL, 'd'},
            {"start_index", required_argument, NULL, 's'},
            {"no_logs", optional_argument, NULL, 'g'},
            {"help", optional_argument, NULL, 'h'},
            {"demo", optional_argument, NULL, 'm'},
            {NULL, 0, 0, '\0'}
    };

    while((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1){
        switch (opt) {
            case 'h':
                usage();
                exit(0);
            case 'o':
                output_path = optarg;
                break;
            case 'i':
                dataset_path = optarg;
                break;
            case 'd':
                dataset_name = optarg;
                break;
            case 'g':
                no_logs = true;
                break;
            case 's':
                id = atoi(optarg);
                break;
            case 'm':
                demo();
                exit(0);
            case '?':
                printf("Unknown option: %c\n",(char)optopt);
                usage();
                exit(-1);
        }
    }
    if(argc  < 9){
        std::cout << 11 - argc <<" more args are required." << std::endl;
        usage();
        exit(-1);
    }

    std::cout << "Check your args setting:" << std::endl;
    std::cout << "\toutput_path: " << output_path << std::endl;
    std::cout << "\tinput_path: " << dataset_path << std::endl;
    std::cout << "\tdataset_name: " << dataset_name << std::endl;
    std::cout << "\tstart_index: " << id << std::endl;
    std::cout << "\tno_logs: " << no_logs << std::endl;
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();


    if (access(output_path.c_str(), F_OK) == -1) {
        // If the directory does not exist, create it
        if (mkdir(output_path.c_str(), 0777) == -1) {
            std::cerr << "Error creating output directory: " << output_path << std::endl;
            exit(EXIT_FAILURE);
        }
    }



    return EXIT_SUCCESS;
}