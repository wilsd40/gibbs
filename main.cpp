#include <iostream>
#include "matrix.h"
#include <map>
#include <string>
#include <vector>
#include <random>
#include <limits>
#include <fstream>
#include <sstream>
#include <numeric>

const int inf = std::numeric_limits<int>::max();

template<typename T>
void print(std::vector<T> vec) {
    std::cout << "[ ";
    for (const auto &i: vec){
        std::cout << i << " ";
    }
    std::cout << "]";
}


class RandomNumberGenerator {
public:
    // Constructor with optional seed
    RandomNumberGenerator(unsigned int seed = std::random_device{}()) 
        : rng(seed) {}

    // Generate a random integer in the range [min, max]
    int getRandomInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng);
    }

    // Generate a random floating-point number in the range [min, max)
    double getRandomDouble(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng);
    }
    
    int getRandomDiscrete(std::vector<int> vals, std::vector<double> weights){
        // need to test this
        
        if (vals.size() != weights.size()) {
            
            throw std::runtime_error("Discrete Int error: vals and weights vectors much match in size");
            
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return vals[dist(rng)];
    }

private:
    std::mt19937 rng; // Mersenne Twister random number generator
};

std::pair<std::vector<int>, std::vector<std::string>> readInput(std::string fn){
    // read input from file 
    
    // Open the file
    std::ifstream infile(fn);
    if (!infile) {

        throw std::runtime_error("Could not open the file!");
    }
    std::vector<int> intvector;
    std::vector<std::string> stringvector;
    std::pair<std::vector<int>, std::vector<std::string>> retpair;
    std::string line;
    int X, Y;

    // Read the first line and parse X and Y
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        if (!(iss >> X >> Y)) {
            throw std::runtime_error("Read error integers");
        }
        intvector.push_back(X);
        intvector.push_back(Y);
        
    }
    

    // Read the second line and store the sequences in a vector<string>
   
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string sequence;
        while (iss >> sequence) {
            stringvector.push_back(sequence);
           
        }
        
        
    }
    else { throw std::runtime_error("Read error strings");
    }
    if ((int)stringvector.size() != intvector.at(1)){
        std::cout << "Did not read " << intvector.at(0) << " strings" << std::endl;
        std::cout << "Read " << (int)stringvector.size() << " strings" << std::endl;
    }

    // Close the file
    infile.close();

    retpair = std::make_pair(intvector, stringvector);

    return retpair;
}

std::pair<std::vector<int>, std::vector<std::string>> readGibbsInput(std::string fn){
    // read input from file 
    
    // Open the file
    std::ifstream infile(fn);
    if (!infile) {

        throw std::runtime_error("Could not open the file!");
    }
    std::vector<int> intvector;
    std::vector<std::string> stringvector;
    std::pair<std::vector<int>, std::vector<std::string>> retpair;
    std::string line;
    int X, Y, Z;

    // Read the first line and parse X and Y
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        if (!(iss >> X >> Y >> Z)) {
            throw std::runtime_error("Read error integers");
        }
        intvector.push_back(X);
        intvector.push_back(Y);
        intvector.push_back(Z);
        
    }
    

    // Read the second line and store the sequences in a vector<string>
   
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string sequence;
        while (iss >> sequence) {
            stringvector.push_back(sequence);
             std::cout << " length = " <<sequence.size() << std::endl;
        }
        
        
    }
    else { throw std::runtime_error("Read error strings");
    }
    if ((int)stringvector.size() != intvector.at(1)){
        std::cout << "Did not read " << intvector.at(0) << " strings" << std::endl;
        std::cout << "Read " << (int)stringvector.size() << " strings" << std::endl;
    }

    // Close the file
    infile.close();

    retpair = std::make_pair(intvector, stringvector);

    return retpair;
}








// before processing the array, make sure all characters are 'A', 'C', 'G', 'T'
// do this in the constructor

Matrix<int> makeCounts(Matrix<char> charray, bool pseudo = false) {
    // count occurences of A, C, G, T in a character matrix
    // each row in charray will be a motif or a string.
    // iterate, row by row, column-by-column, through charray
    // update entry according to key, 
    std::unordered_map<char, int> dna;
    int ncol = charray.get_cols();
    int nrow = charray.get_rows();
    Matrix<int> retmatrix(4, ncol); // this will be returned.
    if (!pseudo) {
    retmatrix.fill(0);
    }
    else {
        retmatrix.fill(1);
    }
    dna['A'] = 0; // row # of rematrix corresponding to 'A' count.
    dna['C'] = 1;
    dna['G'] = 2;
    dna['T'] = 3;
    // iterate, row by row and column by column.  will count occurences of 
    
    for (int rw {0}; rw < nrow; ++rw) {
        for (int cl {0}; cl < ncol; ++cl) {
            int target_row = dna[charray.get(rw, cl)];
            
            retmatrix.set(target_row, cl, (retmatrix.get(target_row, cl) + 1));
        }
    }
    
    return retmatrix;
    
};
    
Matrix<char> stringsToMatrix(std::vector<std::string> stringlist) {
    int nrows {(int)stringlist.size()};
    int ncols {(int)stringlist.at(0).size()};

    Matrix<char> retmatrix(nrows, ncols);
    for (int rw {0}; rw < nrows; ++rw){ 

        if (ncols != (int)stringlist.at(rw).size()){
            throw std::invalid_argument("Invalid string dimensions: all strings must be the same length");
        }
        for (int cl {0}; cl < ncols; ++cl) {
            char setchar {stringlist.at(rw).at(cl)};
            retmatrix.set(rw, cl, setchar);
            
        }
    }
    return retmatrix;
    
}
Matrix<double> doubleToMatrix(std::vector<std::vector<double>> doublelist) {
    int nrows {(int)doublelist.size()};
    int ncols {(int)doublelist.at(0).size()};
    Matrix<double> retmatrix(nrows, ncols);
    for (int rw {0}; rw < nrows; ++rw){ 

        if (ncols != (int)doublelist.at(rw).size()){
            throw std::invalid_argument("Invalid string dimensions: all strings must be the same length");
        }
        for (int cl {0}; cl < ncols; ++cl) {
            double setval {doublelist.at(rw).at(cl)};
            retmatrix.set(rw, cl, setval);
            
        }
    }
    return retmatrix;
    
}

Matrix<double> matrixIntToDouble(Matrix<int> startmat) {
    int nrows {startmat.get_rows()};
    int ncols {startmat.get_cols()};
    Matrix<double> retmat(nrows, ncols);
    for (int rw {0}; rw < nrows; ++rw){
        for (int cl {0}; cl < ncols; ++cl) {
    
    
        double val = static_cast<double>(startmat.get(rw, cl));
        retmat.set(rw, cl, val);
        }
    }
    return retmat;
}

Matrix<double> standardize(Matrix<int> startmat) {
    // reconsider this ... could just make a matrix<double>, and add counts to it instead of using matrix<int> and converting it
    auto dmat = matrixIntToDouble(startmat);
    int ncols = startmat.get_cols();
    int nrows = startmat.get_rows();
    //int lastsum {0};
    for (int cl {0}; cl < ncols; ++cl) { // have to go rows first since summing by column.
        int currsum {0};
        for (int rw {0}; rw < nrows; ++rw) {  // get sum
            currsum = currsum + startmat.get(rw, cl);
        }
        for (int rw {0}; rw < nrows; ++rw) {
            auto cell = dmat.get(rw, cl);
            if (currsum != 0) { // this shouldn't be necessary but avoid divide by 0
                dmat.set(rw, cl, cell/currsum); // divide cell counts by sum
            }
        }
        //lastsum = currsum;
    }
    return dmat;
}

Matrix<double> makeProfile(Matrix<char> startmat) {
    auto countmat = makeCounts(startmat, true); 
    auto retmat = standardize(countmat);
    return retmat;
}

    


int score(Matrix<int> countmat) {
    std::vector<int> results;
    //std::unordered_map dna;
    int rowcount {countmat.get_rows()};
    int colcount {countmat.get_cols()};
    int finalcount {0};
    int themax {0};
    for (int cl {0}; cl < colcount; ++cl){   // note, have to go col by col first because scoring in this manner
        int maxcount = 0;
        int sumcount = 0;
        for (int rw {0}; rw < rowcount; ++rw) {
            int cellcount = countmat.get(rw, cl);
            if (cellcount > maxcount) {
                maxcount = cellcount;
            }
            sumcount += cellcount;
        }
        themax = sumcount - maxcount; //score for this iteration
        results.push_back(themax);
        finalcount += themax;
    }

    return finalcount;
        
}
int scoreit(Matrix<char> charray) {
    auto countmat = makeCounts(charray, false);
    auto retval = score(countmat);
    return retval;
}

Matrix<char> mostProbable(Matrix<char> charray, Matrix<double> profile, int ks) {
    std::unordered_map<char, int> dna;
    if (charray.get_rows() > 1) {
          throw std::out_of_range("mostProbable function: charray must have only 1 row.");
    }
    std::string restring {""};
    dna['A'] = 0;
    dna['C'] = 1;
    dna['G'] = 2;
    dna['T'] = 3;
    double maxprob = 0;
    int pos = 0;
    int charcols {charray.get_cols()};
    for (int i {0}; i < (charcols - ks + 1); ++i) {
        double prob {1};
        for (int j {0}; j < ks; ++j){
            prob = prob * profile.get(dna[charray.get(0, i + j)], j);
        }
        if (prob > maxprob) {            
            maxprob = prob;
            pos = i;
        }
    }

    for (int i {0}; i < ks; ++i) {
        restring += charray.get(0, pos + i);
    }

    Matrix<char> retmatrix(1, ks);
    for (int i {0}; i < ks; ++i){
        auto setval = charray.get(0, pos + i);

        retmatrix.set(0, i, setval);
    }
    return retmatrix;
}

Matrix<char> allMostProbable(Matrix<char> dnas, Matrix<double> profile, int ks) {
    auto dnarows = dnas.get_rows();

    Matrix<char> retmatrix(dnarows, ks);
    retmatrix.fill(' ');
    for (int rw {0}; rw < dnarows; ++rw) {
        auto sendcol = dnas.get_row(rw);    // get string (represented as an entry in a char matrix)
        auto retval = mostProbable(sendcol, profile, ks); //f find most probably kmer
        retmatrix.set_row(rw, retval);
    }
    return retmatrix;
}

Matrix<char> makeRandomMotifs(Matrix<char> dnas, int ks) {
    auto ra = RandomNumberGenerator();
    auto dnacols = dnas.get_cols();
    auto newrows = dnas.get_rows();
    int newcols = ks;
    Matrix<char> retmat(newrows, newcols);
    int lo {0};
    int hi {dnacols - ks};

    for (int rw {0}; rw < newrows ; ++rw) {
        auto val = ra.getRandomInt(lo, hi);
        for (int cl {0}; cl < ks; ++ cl) {
            retmat.set(rw, cl, dnas.get(rw, val + cl));
        }
    
    }
    return retmat;
}
Matrix<char> runRandom(Matrix<char> dnas, int ks) {
    // return value
    int dnarows {dnas.get_rows()};
    // for temporary usage
    
    Matrix<char> retval(dnarows, ks);
    Matrix<double> profile = Matrix<double>(dnarows, ks);
    // first, initialize with a random motif from dnas
    Matrix<char> motifs = makeRandomMotifs(dnas, ks);
    Matrix<char> bestmotifs = motifs; // initial
    
    while (true) {
        profile = makeProfile(motifs);
        motifs = allMostProbable(dnas, profile, ks);
        if (scoreit(motifs) < scoreit(bestmotifs)) { // somehow I lost this line
            bestmotifs = motifs;
        }
        else {
            return bestmotifs;
        }
    }
}
Matrix<char> repeatRandom(Matrix<char> dnas, int ks, int numtries) {
    //int bestscore = inf;
    auto bestmotifs = makeRandomMotifs(dnas, ks);
    for (int i {0}; i < numtries; ++i) {
        auto atrial = runRandom(dnas, ks);
        if (scoreit(atrial) < scoreit(bestmotifs)) {
            bestmotifs = atrial;
        }
        
    }
    return bestmotifs;
}

std::vector<double> probvec(Matrix<char> charray, Matrix<double> profile, int ks) {
    // return vector of probabilities of each kmer
    std::unordered_map<char, int> dna;
    if (charray.get_rows() > 1) {
          throw std::out_of_range("probvec function: charray must have only 1 row.");
    }
    int charcols {charray.get_cols()};
    int numcols = charcols - ks + 1; // number of kmers in charray
    std::vector<double> resvec;
    resvec.reserve(numcols);

    std::string restring {""};
    dna['A'] = 0;
    dna['C'] = 1;
    dna['G'] = 2;
    dna['T'] = 3;

   
    for (int i {0}; i < (charcols - ks + 1); ++i) {
        double prob {1};
        //std::string tempstring {""};
        for (int j {0}; j < ks; ++j){
            prob = prob * profile.get(dna[charray.get(0, i + j)], j);
            //tempstring += charray.get(0, i + j);
        }
        resvec.push_back(prob); // probability of kmer
    
    }
    
    return resvec;
}

Matrix<char> chooseRandomGibbs(Matrix<char> charray, Matrix<double> profile, int row, int ks, RandomNumberGenerator rn) {  
    // pick a motif from row 'row' out of charray based on discrete random distribution.
    // note, when calling this, 'profile' should have row 'row' deleted
    // rn is passed to avoid seeding it; it should be instantiated in main loop
    Matrix<char> cur_row = charray.get_row(row);
    auto probs_vec = probvec(cur_row, profile, ks);
    int probs_size = (int)probs_vec.size();
    
    std::vector<int> indices;
    for (int i {0}; i < probs_size; ++i) {
        indices.push_back(i);
    }

    //std::iota(indices.begin(), indices.end() - 1 + ks, 0);
    //print(indices);
   
    auto selected_index = rn.getRandomDiscrete(indices, probs_vec);
    //std::cout << "picked index " << selected_index << "\n";
    Matrix<char> resmat(1, ks);
    for (int i {0}; i < ks; ++i){
        resmat.set(0, i, cur_row.get(0, selected_index + i));
    }
    
    return resmat;
}


Matrix<char> Gibbs(Matrix<char> dnas, int ks, int N){
    std::vector<int> scores;
    std::vector<Matrix<char>> motiflist;
    auto rn = RandomNumberGenerator();
    auto motifs = makeRandomMotifs(dnas, ks);
    Matrix<char> bestmotif = motifs;
    
    for (int i {0}; i < N; ++i){
        
    
        int ind_motif = rn.getRandomInt(0, dnas.get_rows() - 1);

        Matrix<char> motifs_minus = motifs;
        
        motifs_minus.altdelete_row(ind_motif);

        auto profile = makeProfile(motifs_minus);

        auto newmotif = chooseRandomGibbs(dnas, profile, ind_motif, ks, rn);

        motifs.set_row(ind_motif, newmotif);

        //scores.push_back(scoreit(motifs));
        //motiflist.push_back(motifs);
        if (scoreit(motifs) < scoreit(bestmotif)) {
            bestmotif = motifs;
        }
    
    
    }
    
    //std::cout << "Best motif: " << std::endl;
    //bestmotif.display();
    //std::cout << "Best score: " << scoreit(bestmotif) << std::endl;
    //print(scores);
    

   
    return bestmotif;
    
}

Matrix<char> repeatGibbs(Matrix<char> chmat, int ks, int t, int N) {
    auto best_score = inf;
    auto top_motif = makeRandomMotifs(chmat, ks);
    for (int i {0}; i < t; ++i) {
        std::cout << "on " << i << " trial"<<std::endl;
        auto themotif = Gibbs(chmat, ks, N);
        if (scoreit(themotif) < scoreit(top_motif)) {
            top_motif = themotif;
        }
    }
    top_motif.display();
    std::cout << scoreit(top_motif) << std::endl;
    return top_motif;
}
    
    
    
template<typename T>
void displayResults(Matrix<T> dnas) {
    std::string st {""};    
    for (int i {0}; i < dnas.get_rows(); ++i) {
    
        for (int j {0}; j < dnas.get_cols(); ++j) {
            st += dnas.get(i, j);
        }
        st += " ";
    
    }
    std::cout << st << std::endl;
}



 
    
    



int main(){


//std::cout << dat.first.at(0);
std::string filen {"/home/dan/Downloads/dataset_30309_11 (7).txt"};
auto inpdat = readGibbsInput(filen);
int ks = inpdat.first[0];
int N = inpdat.first[2];
auto dnas = stringsToMatrix(inpdat.second);
std::cout << ks << " " << N << " " << std::endl;
std::cout << dnas.get_rows() << " " << dnas.get_cols() << std::endl;

auto res = repeatGibbs(dnas, ks, 20, N);
displayResults(res);
/*
std::vector<std::string> adna {"TTACCTTAAC",
                          "GATGTCTGTC",
                          "CCGGCGTTAG",
                          "CACTAACGAG",
"CGTCAGAGGT"};
//};                                          
                                  
std::vector<std::string> testit {"CGCCCCTCTCGGGGGTGTTCAGTAACCGGCCA", 
              "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG", 
              "TAGTACCGAGACCGAAAGAAGTATACAGGCGT",
              "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC",
                "AATCCACCAGCTCCACGTGCAATGTTGGCCTA"};



std::vector<std::string> gtest {"CGCCCCTCTCGGGGGTGTTCAGTAACCGGCCA",
 "GGGCGAGGTATGTGTAAGTGCCAAGGTGCCAG",
 "TAGTACCGAGACCGAAAGAAGTATACAGGCGT",
 "TAGATCAAGTTTCAGGTGCACGTCGGTGAACC",
"AATCCACCAGCTCCACGTGCAATGTTGGCCTA"};

std::vector<std::string> mots {"TAAC",
                               "GTCT",
                               //"CCGG",
                               "ACTA",
                               "AGGT"};*/
                               
//auto dnamat = stringsToMatrix(gtest);
//auto motmat =stringsToMatrix(mots);
//auto stmat = stringsToMatrix(dat.second);
//auto motcounts = makeProfile(motmat);
//int ks = 4;
//auto ress = repeatGibbs(dnamat, 8, 20, 100);
/*auto results = probvec(dnamat.get_row(2), motcounts, 4);
print(results);
int ks = 4;
auto ch = dnamat;
auto pr = motcounts;
int rw = 1;
RandomNumberGenerator rn;
std::cout << "Made it here\n";
auto matres = chooseRandomGibbs(ch, pr, rw, ks, rn);
dnamat.get_row(1).display();
matres.display();*/
//results.second.display();
//auto motprobs = standardize(motcounts);
//auto mprob = allMostProbable(stmat, motprobs, 4);
//mprob.display();

//for (int i {0}; i<20; ++i) {
//    auto rndmotifs = makeRandomMotifs(stmat, 4);
    //rndmotifs.display();

//mprob.display();
//auto sc = scoreit(mprob);
//std::cout << sc << std::endl;

//TCTCGGGG CCAAGGTG TACAGGCG TTCAGGTG TCCACGTG
//std::cout << "\n";
//std::cout << "dat[1] = "<<dat.first[1] << std::endl;

//auto dat = readInput("/home/dan/Downloads/dataset_30307_5 (12).txt");
//std::cout << "Working..." << std::endl;
//int ksize = dat.first[1];
//std::vector<std::string> sequences = dat.second;
//Matrix<char> charmatrix = stringsToMatrix(dat.second);
//auto res = repeatRandom(charmatrix, ksize, 1000);
//res.display();
//std::cout << scoreit(res) << std::endl;

//displayResults(res);





    







return 0;
}

