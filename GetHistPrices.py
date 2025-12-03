# Copyright 2019 Gehtsoft USA LLC

# Licensed under the license derived from the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

# http://fxcodebase.com/licenses/open-source/license.html

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import pandas as pd
from config_loader import ConfigLoader
from forexconnect import ForexConnect, fxcorepy
import common_samples

timeframe_minutes = {
    "m1": "00001",
    "m5": "00005",
    "m15": "00015",
    "m30": "00030",
    "H1": "00060",
    "H4": "00240",
    "D1": "01440",
    "W1": "10080",
    "M1": "43200",
}
def parse_args():
    parser = argparse.ArgumentParser(description='Process command parameters.')
    common_samples.add_main_arguments(parser)
    common_samples.add_instrument_timeframe_arguments(parser)
    common_samples.add_date_arguments(parser)
    common_samples.add_max_bars_arguments(parser)
    parser.add_argument('-o', '--output',
                        metavar="OUTPUT_FOLDER",
                        required=True,
                        help='Output folder path for CSV file.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    creds = ConfigLoader.load()
    print("Credentials loaded from config.ini")
    str_user_id = creds['login']
    str_password = creds['password']
    str_url = creds['url']
    str_connection = creds['connection']
    str_session_id = args.session
    str_pin = args.pin
    str_instrument = args.i
    str_timeframe = args.timeframe
    quotes_count = args.quotescount
    date_from = args.datefrom
    date_to = args.dateto
    output_folder = args.output

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate filename: instrument_timeframe_minutes.csv
    instrument_clean = str_instrument.replace('/', '')
    minutes = timeframe_minutes.get(str_timeframe, str_timeframe)
    csv_filename = f"{instrument_clean}_{minutes}.csv"
    csv_path = os.path.join(output_folder, csv_filename)

    with ForexConnect() as fx:
        try:
            fx.login(str_user_id, str_password, str_url,
                     str_connection, str_session_id, str_pin,
                     common_samples.session_status_changed)

            print("")
            print("Requesting a price history...")
            history = fx.get_history(str_instrument, str_timeframe, date_from, date_to, quotes_count)
            current_unit, _ = ForexConnect.parse_timeframe(str_timeframe)
           
            date_format = '%Y.%m.%d %H:%M:%S'
            
            # Convert history to pandas DataFrame and write to CSV
            if current_unit == fxcorepy.O2GTimeFrameUnit.TICK:
                data = []
                for row in history:
                    data.append({
                        'DateTime': pd.to_datetime(str(row['Date'])).strftime(date_format),
                        'Bid': round(row['Bid'], 5),
                        'Ask': round(row['Ask'], 5)
                    })
                df = pd.DataFrame(data, columns=['DateTime', 'Bid', 'Ask'])
                df = df.sort_values('DateTime', ascending=False)
                df.to_csv(csv_path, index=False)
                print(f"Tick data written to: {csv_path}")
                print(f"Records: {len(df)}")
            else:
                data = []
                for row in history:
                    data.append({
                        'DateTime': pd.to_datetime(str(row['Date'])).strftime(date_format),
                        'Open': round(row['BidOpen'], 5),
                        'High': round(row['BidHigh'], 5),
                        'Low': round(row['BidLow'], 5),
                        'Close': round(row['BidClose'], 5),
                        'Volume': row['Volume']
                    })
                df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df = df.sort_values('DateTime', ascending=False)
                df.to_csv(csv_path, index=False)
                print(f"Price history written to: {csv_path}")
                print(f"Records: {len(df)}")
        except Exception as e:
            common_samples.print_exception(e)
        try:
            fx.logout()
        except Exception as e:
            common_samples.print_exception(e)


if __name__ == "__main__":
    main()
