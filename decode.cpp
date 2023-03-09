#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// #include <Eigen/dense>
// #include <quadmath.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
// #include "../../Rcpp11/inst/include/Rcpp/Array.h"


using namespace Rcpp;
using namespace Eigen;

template <typename D> class Decoder {

    // static int last_k;
    // static Vector<D, Dynamic> last_p;

public:
    static NumericVector normal(NumericMatrix logits);
    static NumericVector wu2(NumericMatrix logits, bool verbose = false);
    static NumericVector stratified(NumericMatrix logit, bool verbose = false);
    static D _exp(D x); 
};

template<> double Decoder<double>::_exp(double x) {
    return(exp(x));
}

template<> long double Decoder<long double>::_exp(long double x) {
    return(expl(x));
}

template <typename D>
NumericVector Decoder<D>::wu2(NumericMatrix logits, bool verbose) 
{
    int k = logits.nrow();
    Eigen::Matrix<D, Dynamic, Dynamic> Q(k, k);
    Eigen::Matrix<D, Dynamic, Dynamic> Q1(k + 1, k + 1);
    Eigen::Matrix<D, Dynamic, Dynamic> r(k , k);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i == j)
                r(i, j) = 0;
            else
                r(i, j) = 1 / (1 + _exp(-logits(i, j)));
        }
    }
    // std::cout << r << std::endl;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (i == j) {
                D s = 0;
                for (int m = 0; m < k; m++) {
                    s += r(m, i) * r(m, i);
                }
                Q(i,j) = s;
                Q1(i,j) = s;

            } else {
                Q(i,j) = -r(i,j) * r(j, i);
                Q1(i,j) = Q(i,j);
            }
        }
        Q1(k, i) = 1;
        Q1(i, k) = 1;
    }
    Q1(k, k) = 0;

    Eigen::Matrix<D, Dynamic, 1> v(k + 1, 1);
    for ( int i = 0; i < k; i++) {
        v(i, 0) = 0;
    }
    v(k, 0) = 1;
    Eigen::Matrix<D, Dynamic, 1> p1 = 
        Q1.colPivHouseholderQr().solve(v);
    // Eigen::Matrix<D, Dynamic, 1> p2 = p1 / p1.sum();
    D delta1, delta = (Q1 * p1 - v).squaredNorm();
    int iter = 0;

    do {
        Eigen::Matrix<D, Dynamic, 1> p2(k, 1); 
        Eigen::Matrix<D, Dynamic, 1> p3(k + 1, 1); 
        Eigen::Matrix<D, Dynamic, 1> p = p1.head(k);
        D newb = p.transpose() * (Q * p);
        for (int t = 0; t < k; t++) {
            D s = newb;
            D c = 0;
            for (int j = 0; j < k; j++) {
                if (t == j) {
                    continue;
                }
                // Kahan summation (after wiki pseudocode) 
                D summand = - Q(t, j) * p1(j,0);
                D y = summand - c;
                D t = s + y;
                c = (t - s) - y;
                s = t;
            }
            p2(t, 0) = s / Q(t, t);
        }
        // Eigen::Matrix<D, Dynamic, 1> p4 = p3 / p3.sum();
        p3(k, 0) = -newb;
        D s = p2.sum();
        for (int i = 0; i < k; i++) {
            p3(i,0) = p2(i,0) / s;
        }
        delta1 = (Q1 * p3 - v).squaredNorm();
        if (delta1 >= delta) {
            // printf("%lf %lf\n", delta1, delta);
            if (verbose) {
            }
            // std::cout << delta1 << " " << delta << std::endl;
            break;
        }
        p1 = p3;
        delta = delta1;
        iter++;
    } while(iter < 10);

    if (verbose) {
        std::cout << iter << " extra iterations in wu " << std::endl;
        std::cout << "Remaining delta " << delta <<  std::endl;
        std::cout << "Last delta1 " << delta1 <<  std::endl;
    }


    NumericVector res2(k);
    for (int i = 0; i < k; i++) {
        res2(i) = static_cast<double>(p1(i,0));
    }
    return(res2);
}

template <typename D>
NumericVector Decoder<D>::stratified(NumericMatrix logits, bool verbose) 
{
    int k = logits.nrow();
    Eigen::Matrix<D, Dynamic, Dynamic> Q(k, k);
    Eigen::Matrix<D, Dynamic, Dynamic> Q1(k + 1, k);

    for (int i = 0; i < k; i++) {
        D s = 0;
        for (int j = 0; j < k; j++) {
            if (j != i) {
                s = s + _exp(logits(j, i));
            }
        }
        D pmain = 1 / ( 1 + s);
        for (int j = 0; j < k; j++) {
            if (i == j) {
                Q(j,i) = pmain;
                Q1(j,i) = pmain - 1;
            } else {
                Q(j,i) = pmain * _exp(logits(j, i));
                Q1(j,i) = Q(j, i);
            }
        }
    }
    // std::cout << Q << std::endl;

    Eigen::Matrix<D, Dynamic, 1> b(k + 1, 1);
    for ( int i = 0; i < k; i++) {
        b(i, 0) = 0;
        Q1(k, i) = 1;
    }
    b(k, 0) = 1;
    // std::cout << Q1 << std::endl;
    // std::cout << b << std::endl;

    Eigen::Matrix<D, Dynamic, 1> p1 = 
        Q1.colPivHouseholderQr().solve(b);
    Eigen::Matrix<D, Dynamic, 1> p2 = p1 / p1.sum();

    D delta1, delta = (Q * p2 - p2).squaredNorm();
    int iter = 0;

    do {
       Eigen::Matrix<D, Dynamic, 1> p3 = Q * p2;
       Eigen::Matrix<D, Dynamic, 1> p4 = p3 / p3.sum();
       delta1 = (Q * p4 - p4).squaredNorm();
       if (delta1 >= delta) {
           if (verbose) {
           }
           break;
       }
       iter++;
       p2 = p4;
       delta = delta1;
    } while(1);

    if (verbose) {
        std::cout << iter << " extra iterations" << std::endl;
        std::cout << "Remaining delta " << delta <<  std::endl;
        std::cout << "Last delta1 " << delta1 <<  std::endl;
    }
    
    NumericVector res2(k);
    for (int i = 0; i < k; i++) {
        res2(i) = static_cast<double>(p2(i,0));
    }
    return(res2);
}

#include <map>

using namespace std;

template <typename D> 
NumericVector Decoder<D>::normal(NumericMatrix logits) 
{
    int k = logits.nrow();
    int k1 = k - 1;
    int choose_k2 = k * (k - 1) / 2;
    assert(k == logits.ncol());
    Eigen::Matrix<D, Dynamic, Dynamic> M(k - 1, k * (k - 1) /2);
    Eigen::Matrix<D, Dynamic, 1> v(choose_k2, 1);

    // Initialize first k columns
    //
    for (int i = 0; i < k1; i++) {
        for (int j = 0; j < k1; j++) {
            M(i,j) = i == j ? 1 : 0;
        }
    }

    for (int i = 0; i < k1; i++) {
        v(i, 0) = -logits(0, i + 1);
    }

    int column = k - 1;
    for (int i = 1; i < k1; i++) {
        for (int j = (i + 1); j < k ; j++) {
            for (int m = 0; m < k1; m++) {
                M(m, column) = 0;
            }
            M(i - 1, column) = -1;
            M(j - 1, column) = 1;
            // printf("logits[%d, %d] = %f\n", i,j, logits(i,j));
            v(column, 0) = -logits(i,j);
            column++;
        }
    }
    // cout << M << endl;
    // cout << "Vector" << v << endl;
    // std::cout << M << std::endl;
    // std::cout << v << std::endl;

    Eigen::Matrix<D, Dynamic, Dynamic> Minv(k1, k1);
    for (int i = 0; i < k1; i++) {
        for (int j = 0; j < k1; j++) {
            Minv(i,j) = ((i == j) ? 2 : 1) / static_cast<D>(k);
        }
    }
    // std::cout << Minv << std::endl;

    Eigen::Matrix<D, Dynamic, 1> res = Minv * (M * v);
    // std::cout << res << std::endl;
    Eigen::Matrix<D, Dynamic, 1> r(k, 1);
    r(0, 0) = 1;
    for (int i = 0; i < k1; i ++) {
        r(i + 1, 0) = _exp(res(i, 0));
    }
    // std::cout << r << std::endl;
    D s = r.sum();
    // std::cout << s << std::endl;
    NumericVector res2(k);
    for (int i = 0; i < k; i++) {
        res2(i) = static_cast<double>(r(i, 0) / s);
    }
    return(res2);
}

// [[Rcpp::export]]
//
NumericVector normal_d(NumericMatrix logits) {
    return Decoder<double>::normal(logits);
}

// [[Rcpp::export]]
//
NumericVector normal_ld(NumericMatrix logits) {
    return Decoder<long double>::normal(logits);
}

// [[Rcpp::export]]
//
NumericVector stratified_d(NumericMatrix logits, bool verbose = false) {
    return Decoder<double>::stratified(logits, verbose);
}

// [[Rcpp::export]]
//
NumericVector stratified_ld(NumericMatrix logits, bool verbose = false) {
    return Decoder<long double>::stratified(logits, verbose);
}

// [[Rcpp::export]]
//
NumericVector wu2_d(NumericMatrix logits, bool verbose = false) {
    return Decoder<double>::wu2(logits, verbose);
}

// [[Rcpp::export]]
//
NumericVector wu2_ld(NumericMatrix logits, bool verbose = false) {
    return Decoder<long double>::wu2(logits, verbose);
}
