rm(list = ls()) ; gc()
require(glmnet)
require(TunePareto)
require(doMC)
require(data.table)

test_cut = "2019-01-01"
test_end = "2020-01-01"
data_path = "/media/harunkivril/HDD/MsThesis/csv_data"
prod_file_name = "/eligible_production.csv"
#prod_file_name = "/aliaga_prod.csv"
production = fread(paste0(data_path, prod_file_name))
three_hourly = FALSE
monthly_results = FALSE

test_weather = function(train_data, test_data, name) {
    merged_cols = colnames(train_data)
    cond = paste0(name, "_WS|", name, "_WDIR")
    train_cols = merged_cols[grepl(cond, merged_cols)]

    X_train = train_data[, train_cols, with=F]
    y_train = train_data$production
    X_test = test_data[, train_cols, with=F]
    y_test = test_data$production

    cvid = generateCVRuns(
        wday(train_data$date), 1, nfold = 5, stratified = TRUE)
    a = rbindlist(
        lapply(c(1 : length(cvid[[1]])),
        function(x) data.table(fid = x, insid = cvid[[1]][[x]]))
    )
    foldids = a[order(insid)]$fid
    registerDoMC(10)
    glmnet.control(devmax=0.999,mxit=100)

    print("Fitting Model...")
    model_fit = cv.glmnet(
        as.matrix(X_train),
        y_train, #Change here
        family = "gaussian",
        alpha=1,
        parallel=TRUE,
        nlambda=50,
        foldid=foldids,
        intercept=TRUE,
        type.measure="mae",
        standardize=TRUE
    )
    print("Done Fitting.")

    y_pred = predict(model_fit, as.matrix(X_test), s = "lambda.min")
    wmape = sum(abs(y_pred - y_test))/sum(y_test)
    print(paste(name ,"WMAPE", wmape))
    test_results = test_data[, c("date", "hour", "production"), with=F]
    test_results[, pred := y_pred]
    if (monthly_results){
        test_results[, month := lubridate::month(date)]
    } else {
       test_results[, month := "all"]
    }
    results_sum = (test_results[, .(
        wmape = sum(abs(pred-production))/sum(production)
        #bias=sum(pred-production)/sum(production)
        ), by="month"])

    return(results_sum)

}


all_results = data.table()
for (farm_eic in unique(production$eic)){
    era5_features = fread(paste0(data_path, "/", farm_eic, "/era5_pres.csv.gz"))
    gefs_features = fread(paste0(data_path, "/", farm_eic, "/gefs_pres.csv.gz"))
    pred_features = fread(paste0(data_path, "/", farm_eic, "/pred_pres.csv.gz"))
    #gfs_features = fread(paste0(data_path, "/gfs_pres.csv.gz"))

    merged = merge(era5_features, gefs_features)
    merged = merge(merged, pred_features)
    #merged = merge(merged, gfs_features)
    merged = merge(merged, production[eic == farm_eic])

    if (three_hourly) {
        merged = merged[hour %% 3 == 0]
    }

    train_data = merged[date < test_cut]
    test_data = merged[date >= test_cut & date < test_end]

    era5_res = test_weather(train_data, test_data, "ERA5")
    gefs_res = test_weather(train_data, test_data, "GEFS")
    #gfs_res = test_weather(train_data, test_data, "GFS")
    pred_res = test_weather(train_data, test_data, "PRED")

    merged_results = merge(era5_res, gefs_res, by="month", suffixes = c(".era5", ".gefs"))
    merged_results2 = merge(gefs_res, pred_res, by="month", suffixes = c(".gefs", ".pred"))
    merged_results = merge(merged_results, merged_results2, by=c("month", "wmape.gefs"))
    merged_results[, eic:=farm_eic]

    print(merged_results)

    all_results = rbind(all_results, merged_results)
}

all_results
