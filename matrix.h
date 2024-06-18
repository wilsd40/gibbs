#ifndef MATR_H
#define MATR_H
#include <memory>
#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>

template <typename T> class Matrix {

public:
    //using value_type = int; // The type of the elements
    using difference_type = std::ptrdiff_t;
    using pointer = std::unique_ptr<T[]>;
    using reference = T&;
    using raw_pointer = T*;
/*    class Iterator {

    public:
        using iterator_category = std::forward_iterator_tag;

        
        Iterator(pointer&, size_t);
        Iterator(const Iterator& other);  
        Iterator& operator++();
        Iterator operator++(int);
        Iterator& operator--();
        Iterator operator--(int);
        Iterator& operator=(const Iterator&);
        //(no need to write copy constructor since shallow copy) need to write a copy constructor
        ~Iterator()=default;
        
        reference operator*(); // in cpp file, needs to be Matrix::reference
        raw_pointer get_address();
        raw_pointer operator->();
        friend value_type operator-(const Iterator, const Iterator);
        friend bool operator==(const Iterator, const Iterator);
        friend bool operator!=(const Iterator, const Iterator);
        
    private:
                
        pointer& datptr;
        size_t index;
       
        
    };*/
    void fill(T fillval){ // i put this before the constructor but it might be ok after the constructor

        for (int rw {0}; rw < rows; ++rw){
            for (int cl {0}; cl < cols; ++cl){
                matptr[offset(rw, cl, true)] = fillval;
            }
        } 

    }
    
    //constructor
    Matrix(int numrows, int numcols): rows(numrows), cols (numcols), size (rows * cols), 
    matptr {(numrows>0 && numcols>0) ? new T[rows * cols] : nullptr} {
        if (numrows < 1 || numcols < 1) {
            throw std::invalid_argument("Invalid matrix dimensions: Rows and columns must be greater than zero");
     
        }
        fill(0); // need to fix if using char or others
    } // initialize array to 0s}
    
    //copy constructor
    Matrix(const Matrix& source): rows{source.rows}, cols{source.cols}, size{source.size}, 
    matptr {size ? new T[rows * cols] : nullptr} { // copy constructor
    //deep copy
        std::copy(source.matptr.get(), source.matptr.get() + size, matptr.get());
    //std::cout << "Copy constructor called" << std::endl;
        }
        
    //move constructor
    Matrix(Matrix&& source) noexcept: rows(source.rows), cols(source.cols), size(source.size), matptr(std::move(source.matptr)) { // move constructor
        source.rows = 0;
        source.cols = 0;
        source.size = 0;
    //std::cout << "Move constructor called" << std::endl;

    }

    int get_rows() {
        return rows;
    }

    int get_cols() {
        return cols;
    }
    void set_rows(int rw) {
        this->rows = rw;
    }
    void set_cols(int cl) {
        this->cols = cl;
    }
    
    void display_dims() {
        std::cout << "(" << get_rows() << "," << get_cols() << ")" << std::endl;
    }
    
    void set(int row, int col, T val){
        matptr[offset(row, col)] = val;
    }
    
    T get(int row, int col) {
        return matptr[offset(row, col, true)];
    }
    
    void set_row(int row_num, T val) {
        if (row_num <0 || row_num >= rows) {
            throw std::out_of_range("fill row : index out of range");        
        }
        for (int cl {0}; cl < cols ; ++cl){
            matptr[offset(row_num, cl, true)] = val;
        }
    }
    
    
    void set_row(int row_num, std::vector<T> vals) {

        if (vals.get_cols() != cols || row_num <0 || row_num >= rows) {
            //std::cout << vals.get_cols() << " " << row_num << " " << rows << std::endl;
            throw std::out_of_range("fill row : index out of range");        
        }
        for (int cl {0}; cl < cols ; ++cl){
            matptr[offset(row_num, cl, true)] = vals.at(cl);
        }
    }
    
    void set_row(int row_num, Matrix<T> valMatrix) {

        if ((valMatrix.get_cols() != this->get_cols() || valMatrix.get_rows() != 1) or (row_num < 0 || row_num >= this->get_rows())) {
            //std::cout << valMatrix.get_cols() << " " << row_num << " " << this->get_rows() << std::endl;
            throw std::out_of_range("fill row : index out of range");    
        }
        for (int cl {0}; cl < this->get_cols() ; ++cl){
            //std::cout << row_num << " " << cl << std::endl;
            matptr[offset(row_num, cl, true)] = valMatrix.at(0, cl);
        }
            
    }
    
    Matrix<T> get_row(int row_num) {    // will return a matrix corresponding to a row
        if (row_num < 0 || row_num >= rows) {
            throw std::out_of_range("get row : index out of range");    
        }
        
        Matrix<T> remat(1, this->get_cols());
        for (int cl {0}; cl < this->get_cols(); ++cl) {
            remat.set(0, cl, this->get(row_num, cl));
        }
        return remat;
            
        }   
    
    void set_col(int col_num, T val) {
        if (col_num < 0 || col_num >= cols) {
             throw std::out_of_range("set col : index out of range");    
        }
        for (int rw {0}; rw < this->get_rows(); ++rw) {
            matptr[offset(rw, col_num, true)] = val;
        }
    }
    
    void set_col(int col_num, Matrix valMatrix) {
        if (valMatrix.get_rows() != this->get_rows() || valMatrix.get_cols() != 1 || col_num < 0 || col_num >= this->get_cols()){
  
            throw std::out_of_range("fill col : index out of range");    
        
        }
        for (int rw {0}; rw < this->get_rows(); ++rw) {
            matptr[offset(rw, col_num, true)] = valMatrix.get(rw, 0);
        }
    }
    
    Matrix<T> get_col(int col_num) {
        if (col_num < 0 || col_num >= this->get_cols()) {
            throw std::out_of_range("fill col : index out of range");    
            std::cout << "error here" << std::endl;
        }
        Matrix<T> retMat(this->get_rows(), 1);
        for (int rw {0}; rw < this->get_rows(); ++rw) {
            retMat.set(rw, 0, this->get(rw, col_num));
        }
        return retMat;
    }
    
    void delete_row(int row_num) {
        // calculate size of new array
        if (row_num < 0 || row_num >=this->get_rows()) {
            throw std::out_of_range("row number out of range");
        }
        int newsize {(this->get_rows() - 1) * (this->get_cols())};
        // allocate new memory from heap
        std::unique_ptr<T[]> newpointer(new T[newsize]);
        // deep copy from old array;
        int newcounter {0};

        for (int rw {0}; rw < this->get_rows(); ++rw) {
            //++tot;

            if (rw != row_num) { // skip deleted row.

                for (int cl {0}; cl < this->get_cols(); ++cl) {
                    newpointer[newcounter++] = matptr[offset(rw, cl, true)]; 
                    
                    // this is clunky; there is a better way to do this using a counter for newpointer
                    // to revise it, initialize counter to 0 then use newpointer[counter++]
                    
                    //if (rw < row_num){
                    //    newpointer[offset(rw, cl, true)] = matptr[offset(rw, cl, true)]; // rows before deletion

                    //}
                    //else if (rw > row_num) {
                    //    newpointer[offset(rw - 1, cl, true)] = matptr[offset(rw, cl, true)]; // rows after deletion
                    //}
                }
            }
        }
        this->set_rows(this->get_rows() - 1);
        this->size = this->size - this->get_cols();
        matptr = std::move(newpointer);
        
    }

    
    
    void altdelete_row(int row_num) {
        if (row_num < 0 || row_num >=this->get_rows()) {
            throw std::out_of_range("row number out of range");
        }
        if (!(row_num == (this->rows - 1))){
            std::copy((matptr.get() + (row_num + 1) * cols), matptr.get() + rows * cols , matptr.get() + (row_num) * cols);
        }
        --rows;
        this->size = this->size - this->get_cols();
        
    }

                
    
        
    
    void bounds(int row, int col) { // does bounds checking, throws exception 
        if (!(row >= 0 && row < rows && col >= 0 && col < cols)) {
            throw std::out_of_range("Index out of range");
        }

    }   
    
    int offset(int row, int col) {
    return (row * cols + col); // double check this formula
    }
    
    int offset(int row, int col, bool check) { // 'check' does nothing, just distinguishes from offset.
        bounds(row, col);
        return (row * cols + col); //
    
    }
    bool bool_bounds(int row, int col) { // does bounds checking, returns true 
        if (!(row >= 0 && row < rows && col >= 0 && col < cols)) {
            return false;
        }
        return true;
    }
    
    
    
    T at(int row, int col) {
    return matptr[offset(row, col, true)];
    }
    
    Matrix& operator=(Matrix other){ // overloaded = for copy & swap revision
        swapit(*this, other);
        //std::cout << "overloaded = called" << std::endl;
    return *this;
    }

    T& operator()(int row, int col) {
        return matptr[offset(row, col, true)];
    }
    // need to write delete row.
    
    
    
    //Iterator begin();
    //Iterator end();
    void display() {
    for (int i {0}; i < rows; ++i){
        for (int j {0}; j < cols; ++j){
            std::cout << std::setw(2) << matptr[offset(i,j)] << " ";
        }
        std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    ~Matrix() = default;
    
private:
    int rows;
    int cols;
    int size;
    pointer matptr;
    template <typename F>
    void swapit(Matrix<F>& first, Matrix<F>& second) noexcept {
        std::swap(first.rows, second.rows);
        std::swap(first.cols, second.cols);
        std::swap(first.size, second.size);
        std::swap(first.matptr, second.matptr);
    
    }
    
};


#endif // MATR_H
