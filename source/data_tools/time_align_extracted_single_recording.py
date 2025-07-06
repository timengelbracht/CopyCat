from data_indexer import RecordingIndex
from data_loader_aria import AriaData
from data_loader_iphone import IPhoneData
from data_loader_gripper import GripperData
import os
from pathlib import Path
from time_aligner import TimeAligner
from qrcode_detector_decoder import QRCodeDetectorDecoder
from typing import Dict, List, Tuple
import ast
import json

class Datasyncer:
    def __init__(self, base_path, rec_location, rec_type, interaction_indices, data_indexer):
        self.base_path = base_path
        self.rec_location = rec_location
        self.rec_type = rec_type
        self.interaction_indices = interaction_indices
        self.data_indexer = data_indexer

        self.data_manager = {}

        self.registered = False

    def register_all_data_loaders(self, save_time_sync_info: bool = True):
        """
        This method registers all data loaders for the specified recording modules.
        It iterates through the recording modules and initializes the appropriate data loader.
        """
        queries_at_loc = self.get_all_data_for_loc_and_interaction()

        for loc, inter, rec, ii, path in queries_at_loc:
            print(f"Found recorder: {rec} at {path}")
            rec_module = rec
            self.register_data_loader_for_rec_module(rec_module)

        self.registered = True

        a = 2 


        # After registering all data loaders, tiem deltas of eacxh sensor stream 
        # with respect to aria_human is calculated.
        # get aria_human data loader
        aria_human_data = self.data_manager["aria_human"]
        aria_human_time_pair = aria_human_data["time_pair"]

        manuals = []
        for idx, (rec_module, data_info) in enumerate(self.data_manager.items()):
            data_loader = data_info["data_loader"]

            if data_info["delta"] is not None:
                print(f"[{data_loader.logging_tag}] delta already present."
                    f"({data_info['delta']} ns) skipping.")
                continue

            time_pair = data_info["time_pair"]
    
            # if qr code is not detected manual aligment is done
            if time_pair[0] is None or time_pair[1] is None:
                print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")
                manuals.append((rec_module, data_loader))
                continue

            # if qr code is detected, time delta is calculated
            print(f"[{data_loader.logging_tag}] QR code detected, calculating time delta.")
            time_aligner = TimeAligner(aria_pair=aria_human_time_pair, sensor_pair=time_pair)
            self.data_manager[rec_module]["delta"] = time_aligner.get_delta()
            print(f"[{data_loader.logging_tag}] Time delta: {time_aligner.get_delta()} ns")

        # iterate over manual aligments (it can technicaly only be then gripper)
        for rec_module, data_loader in manuals:
            print(f"[{data_loader.logging_tag}] No QR code detected, using manual time alignment.")

            # read manual time alignment from file
            event_pairs = {}
            path = data_loader.extraction_path / "event_pairs_for_manual_time_alignment.txt"
            if not path.exists():
                raise FileNotFoundError(f"Manual time alignment file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                # --- 1 · header line --------------------------------------------------
                header_line = f.readline().strip()
                header: Tuple[str, str] = ast.literal_eval(header_line)
                event_pairs["sensor_1"] = header[0]
                event_pairs["sensor_2"] = header[1]

                # --- 2 · remaining lines ---------------------------------------------
                pairs: List[Tuple[int, int]] = []
                for line in f:
                    line = line.strip()
                    if not line:                       # skip blank lines
                        continue
                    pair = ast.literal_eval(line)
                    # ensure they’re ints
                    pairs.append((int(pair[0]), int(pair[1])))
                event_pairs["pairs"] = pairs

            # get sens1_to_aria_aligner
            sens1_to_aria_human = TimeAligner.from_delta(
                delta_ns=self.data_manager[event_pairs["sensor_1"]]["delta"]
            )
            sens2_to_sens1 = TimeAligner.from_event_pairs(event_pairs["pairs"])
            sens2_to_aria_human = TimeAligner.chain(
                sens1_to_aria_human, sens2_to_sens1
            )   
            self.data_manager[rec_module]["delta"] = sens2_to_aria_human.get_delta()
            self.data_manager[rec_module]["manual_alignment"] = True
            print(f"[{data_loader.logging_tag}] Manual time delta: {sens2_to_aria_human.get_delta()} ns")

        # save time sync info
        if save_time_sync_info:
            self.save_all_time_sync_info()
        else:
            print("Time sync info not saved. Set save_time_sync_info=True to save it.")

        a = 2

    def save_all_time_sync_info(self):    
        """
        This method saves the time synchronization information for all registered data loaders.
        """
        if not self.registered:
            raise RuntimeError("Data loaders must be registered before saving time sync info.")

        for rec_module, data_info in self.data_manager.items():
            data_loader = data_info["data_loader"]
            time_pair = data_info["time_pair"]
            delta = data_info["delta"]
            manual_alignment = data_info["manual_alignment"]

            # Create directory for the recording module
            file_path = data_loader.extraction_path / "time_sync_info.json"

            if file_path.exists():
                print(f"[{data_loader.logging_tag}] Time sync info already exists at {file_path}.")
                continue

            # Save the time sync info to a JSON file
            time_sync_info = {
                "rec_module": rec_module,
                "time_pair": time_pair,
                "delta": delta,
                "manual_alignment": manual_alignment,
            }
            with file_path.open("w", encoding="utf-8") as f:
                import json
                json.dump(time_sync_info, f, indent=4)
            
            print(f"[{data_loader.logging_tag}] Time sync info saved to {file_path}")

    def get_all_data_for_loc_and_interaction(self):
        """
        This method retrieves all data for a specified location and interaction type.
        It uses the RecordingIndex to query the data based on the provided parameters.
        """
        queries_at_loc = self.data_indexer.query(
            location=self.rec_location,
            interaction=self.rec_type,
            recorder=None,
            interaction_index=self.interaction_indices
        )
        
        return queries_at_loc
    
    def register_data_loader_for_rec_module(self, rec_module: str):
        """
        Retrieves and registers the appropriate data loader based on the recording module type.
        - For 'gripper' the module name must be **exactly** 'gripper'.
        - For all others (e.g. 'aria_human_ego', 'iphone_left') a substring match is enough.
        """
        data_loader_classes = {
            "gripper": GripperData,
            "iphone": IPhoneData,
            "aria": AriaData,
        }

        for key, loader_class in data_loader_classes.items():
            # Exact match required only for 'gripper'
            match = (rec_module == key) if key == "gripper" else (key in rec_module)
            if not match:
                continue

            data_loader = loader_class(
                self.base_path,
                self.rec_location,
                self.rec_type,
                rec_module,
                self.interaction_indices,
                self.data_indexer,
            )

            # check if time alignment is already done
            file_path = data_loader.extraction_path / "time_sync_info.json"
            if file_path.exists():
                print(f"[{data_loader.logging_tag}] Time sync info already exists at {file_path}.")
                with file_path.open("r", encoding="utf-8") as f:
                    time_sync_info = json.load(f)
                    data_loader.time_pair         = tuple(time_sync_info["time_pair"])
                    data_loader.delta             = time_sync_info["delta"]
                    data_loader.manual_alignment  = time_sync_info["manual_alignment"]
                    print(f"[{data_loader.logging_tag}] Loaded time sync info: {time_sync_info}")

                # ←-- register and stop looping
                self.data_manager[rec_module] = {
                    "data_loader":       data_loader,
                    "time_pair":         data_loader.time_pair,
                    "delta":             data_loader.delta,
                    "manual_alignment":  data_loader.manual_alignment,
                }
                break

            self.data_manager[rec_module] = {
                "data_loader": data_loader,
                "time_pair": self._get_time_pair_for_single_recording_module(
                    data_loader, stride=2
                ),
                "delta": None,
                "manual_alignment": False,
            }
            break
        else:
            # Optional: warn if nothing matched
            raise ValueError(f"No data-loader rule matched rec_module='{rec_module}'.")

    def _get_time_pair_for_single_recording_module(self, data_loader, stride: int = 1):
        """
        This method retrieves time pairs (qr code timestamp, recording timestamp) for a single recording module.

        """
        print("###############################################")
        print(f"[{data_loader.logging_tag}] - Extracting time pairs...")
        print("###############################################")
        rgb_dir = Path(data_loader.extraction_path / data_loader.label_rgb.strip("/"))
        rgb_ext = data_loader.rgb_extension

        qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
        device_ts, qr_ts = qr_detector.find_first_valid_qr(stride=stride)
        print(f"{data_loader.logging_tag} - Finished extracting time pairs.")

        return (device_ts, qr_ts)

    def apply_time_deltas_to_data_streams(self, data_loader: str, sensor_ts: int) -> int:
        pass
        #TODO: Implement this method to apply time deltas to data streams.
        

if __name__ == "__main__":

    rec_location = "bedroom_1"
    rec_interaction = "gripper"
    interaction_indices = "1-8"

    base_path = Path(f"/data/ikea_recordings")

    data_indexer = RecordingIndex(
        os.path.join(str(base_path), "raw") 
    )

    # Initialize the Datasyncer with the base path, location, interaction type, and interaction indices
    data_syncer = Datasyncer(
        base_path=base_path,
        rec_location=rec_location,
        rec_type=rec_interaction,
        interaction_indices=interaction_indices,
        data_indexer=data_indexer
    )

    # Register all data loaders for the specified recording modules
    data_syncer.register_all_data_loaders()


    # # get all data for the specified location and interaction
    # queries_at_loc = data_indexer.query(
    #     location=rec_location, 
    #     interaction=rec_interaction, 
    #     recorder=None,
    #     interaction_index=interaction_indices
    # )

    # time_pairs = []
    # for loc, inter, rec, ii, path in queries_at_loc:
    #     print(f"Found recorder: {rec} at {path}")

    #     rec_type = inter
    #     rec_module = rec
    #     interaction_indices = ii

    #     if "gripper" in rec_module:
    #         gripper_data = GripperData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            
    #         a = 2


    #     if "iphone" in rec_module:
    #         iphone_data = IPhoneData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)


    #         a = 2
    #     if "aria" in rec_module:
    #         aria_data = AriaData(base_path, rec_location, rec_type, rec_module, interaction_indices, data_indexer)
            
    #         # get rgb frames
    #         rgb_dir = Path(aria_data.extraction_path / aria_data.label_rgb.strip("/"))
    #         rgb_ext = aria_data.rgb_extension

    #         qr_detector = QRCodeDetectorDecoder(rgb_dir, ext=rgb_ext)
    #         device_ts, qr_ts = qr_detector.find_first_valid_qr()

    #         time_pairs.append(((device_ts, qr_ts), rec_module))
            




    # get all distinct interaction indices
    # interaction_indices_at_loc = set()
    # for loc, inter, rec, ii, path in queries_at_loc:
    #         interaction_indices_at_loc.add(ii)

    

        



    a = 2