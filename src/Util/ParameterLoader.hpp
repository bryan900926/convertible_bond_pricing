#include "../src/Const/ParameterColumn.hpp"
#include "../src/Pricing/CbModel.h"
#include <boost/pfr.hpp>
#include <rapidcsv.h>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#define PARAMETER_CHECK(cond, msg) \
    do { \
        if (!(cond)) [[unlikely]] { \
            std::cerr << "[FATAL PARAM ERROR] " << __FILE__ << ":" << __LINE__ << "\n" \
                      << "Condition Failed: " << #cond << "\n" \
                      << "Details: " << (msg) << "\n" \
                      << "Terminating process.\n"; \
            std::abort(); \
        } \
    } while(false)

template<typename ResultType>
class ParameterLoader {
public:
    enum class FileMode {
        Append,
        Overwrite
    };

    static ParameterLoader StartLoadParams(const std::string &ticker,
                                 const std::string &parameter_csv_path,
                                 const std::string &call_schedule_path) {

      rapidcsv::Document doc(parameter_csv_path, rapidcsv::LabelParams(0, 0),
                             rapidcsv::SeparatorParams(',', true));
        CbParas cb = {.call_schedule = CallSchedule::Create({}, "")};
        CdgParas cdg = {};
        VasciekParas vas = {};

        // Use ssize_t for rapidcsv indexes
        const ssize_t i = doc.GetRowIdx(ticker);
        if (i == -1) {
            throw std::runtime_error("Ticker not found in parameter CSV: " + ticker + 
                                     ". Also, the ticker column needs to be the first column in the CSV.");
        }
        // --- Extract CbParas ---
        cb.T           = doc.GetCell<double>(ParameterColumn::T, i);            // Maturity of the bond in year
        cb.F           = doc.GetCell<double>(ParameterColumn::F, i);            // Face value of the bond
        cb.rr          = doc.GetCell<double>(ParameterColumn::rr, i);           // Recovery rate of the bond
        cb.CR          = doc.GetCell<double>(ParameterColumn::CR, i);           // Conversion ratio of the bond
        cb.NS          = doc.GetCell<double>(ParameterColumn::NS, i);           // Number of shares of the underlying stock
        PARAMETER_CHECK(cb.NS > 0, "Number of shares (NS) must be positive for ticker: " + ticker);
        cb.NC          = doc.GetCell<double>(ParameterColumn::NC, i);           // Number of convertible bonds issued
        cb.CP          = doc.GetCell<double>(ParameterColumn::CP, i);           // Call price of the bond
        cb.rho         = doc.GetCell<double>(ParameterColumn::rho, i);          // Correlation between the interest rate and the company value
        cb.coupon_rate = doc.GetCell<double>(ParameterColumn::coupon_rate, i);  // Coupon rate of the bond
        cb.dt_other    = doc.GetCell<double>(ParameterColumn::dt_other, i);     // Time step for the tree, in years
        PARAMETER_CHECK(cb.dt_other > 0, "Time step (dt_other) must be positive for ticker: " + ticker);
        cb.paid_cycle = doc.GetCell<int>(ParameterColumn::paid_cycle, i);       // Coupon payment cycle, 1 for annual, 0.5 for semi-annual, etc.
        cb.s0 = doc.GetCell<double>(ParameterColumn::s0, i); // Initial stock price
        PARAMETER_CHECK(cb.s0 > 0, "stock price (s0) must be positive for ticker: " + ticker);
        cb.partition = 20; // Number of partitions for the l dimension in the tree
        cb.qdt = 20;

        // --- Extract CDG Model Params ---
        cdg.lamda   = doc.GetCell<double>(ParameterColumn::lamda, i);           // Mean reversion speed of the leverage ratio
        cdg.phi     = doc.GetCell<double>(ParameterColumn::phi, i);             // Sensitivity of the leverage ratio to the interest rate
        cdg.l0      = doc.GetCell<double>(ParameterColumn::l0, i);              // Initial leverage ratio (should be negative)
        PARAMETER_CHECK(cdg.l0 < 0, "l0 must be negative for ticker: " + ticker);
        cdg.V0 = doc.GetCell<double>(ParameterColumn::V0, i);
        PARAMETER_CHECK(cdg.V0 > 0, "V0 must be positive for ticker: " + ticker);
        cdg.delta   = doc.GetCell<double>(ParameterColumn::delta, i);
        cdg.sigma_v = doc.GetCell<double>(ParameterColumn::sigma_v, i);
        PARAMETER_CHECK(cdg.sigma_v > 0, "sigma_v must be positive for ticker: " + ticker);
        cdg.v = doc.GetCell<double>(ParameterColumn::v, i);                     // nu : drift of the leverage ratio

        // --- Extract Vasicek Model Params ---
        vas.kappa   = doc.GetCell<double>(ParameterColumn::kappa, i);
        vas.r_bar   = doc.GetCell<double>(ParameterColumn::r_bar, i);
        vas.sigma_r = doc.GetCell<double>(ParameterColumn::sigma_r, i);
        PARAMETER_CHECK(vas.sigma_r > 0, "sigma_r must be positive for ticker: " + ticker);
        vas.r0      = doc.GetCell<double>(ParameterColumn::r0, i);

        // 2. Removed LabelParams(0, 0) here because multiple rows share the same ticker!
        rapidcsv::Document call_doc(call_schedule_path);
        
        std::vector<CallInfo> call_infos;
        size_t row_count = call_doc.GetRowCount();
        for (size_t r = 0; r < row_count; ++r) {
            std::string tmp_ticker = call_doc.GetCell<std::string>(ParameterColumn::Ticker, r);
            if (tmp_ticker != ticker) {
                continue; // Skip rows that don't match the current ticker
            }
            std::string date = call_doc.GetCell<std::string>(ParameterColumn::CallDate, r);
            double price = call_doc.GetCell<double>(ParameterColumn::CallPrice, r);
            call_infos.emplace_back(date, price);
        }

        std::string pricing_date = doc.GetCell<std::string>(ParameterColumn::PricingDate, i);
        if (pricing_date.size() != 10 || pricing_date[4] != '-' || pricing_date[7] != '-') {
            throw std::runtime_error("Pricing date format is incorrect for ticker: " + ticker + 
                                     ". Expected format: YYYY-MM-DD, got: " + pricing_date);
        }
        cb.call_schedule = CallSchedule::Create(call_infos, pricing_date);
        
        // Pass the ticker to the constructor as well
        return ParameterLoader(cb, cdg, vas, ticker);
    };

    void Save(const std::string& path, FileMode mode = FileMode::Overwrite) {
        auto ios_mode = (mode == FileMode::Overwrite) ? std::ios::trunc : std::ios::app;
        std::ofstream ofs(path, ios_mode);

        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }

        ResultType result = CbTreePricingMemoSave(cb, cdg, vas, ticker);

        if (mode == FileMode::Overwrite) {
            ofs << "Ticker";

            boost::pfr::for_each_field(
                result,
                [&](const auto & field, auto index) {
                  constexpr std::size_t I = decltype(index)::value;
                  constexpr auto name = boost::pfr::get_name<I, ResultType>();
                  ofs << "," << name;
                }
            );
            ofs << "\n";
        }

        ofs << ticker;
        
        boost::pfr::for_each_field(result, [&](const auto& field, auto /*index*/) {
            ofs << "," << field;
        });
        ofs << "\n";
    }

private:
    // 3. Added ticker to constructor and member variables
    ParameterLoader(const CbParas &cb, const CdgParas &cdg, const VasciekParas &vas, const std::string &ticker)
        : cb(cb), cdg(cdg), vas(vas), ticker(ticker) {}

    CbParas cb;
    CdgParas cdg;
    VasciekParas vas;
    std::string ticker;
};