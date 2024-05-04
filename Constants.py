# Common Configuration
symbol = 'maticusdt'
mills_for_1min = 60*1000
vae_model_path = 'VAE_model/vae_matic_model.pth'
pre_processed = 'pre_processed/matic.csv'
min_max_scaler = 'scalar/min_max_scalar.pkl'
standard_scaler = 'scalar/standard_scalar.pkl'
wgan_x_scaler_path = 'trained_gan/wgan_x_scaler.pkl'
wgan_y_scaler_path = 'trained_gan/wgan_y_scaler.pkl'

# 1h, 5 min configuration
# interval = '5m'
# sample_rate_1h = 12
# total_time_steps = 60001
# no_of_points = 5000
# num_components = 17
# total_time_steps = 700
# no_of_points = 400
# sliding_window_size = 10

# 4h, 15 min configuration
interval = '15m'
sample_rate_1h = 16
total_time_steps = 80001
no_of_points = 5000
num_components = 17
total_time_steps = 7000
no_of_points = 400
sliding_window_size = 10