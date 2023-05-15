## Optimal and SubOptimal Decision Trees

**Disclaimer: This is a work In Progress**

Implementation of Less Greedy Decision Trees algorithms + DL8.5  and its extensions using rust and a python wrapper.


A documentation will follow soon.

### Running the code:
- Install [Rust](https://www.rust-lang.org/tools/install) and ensure that cargo is available in the **PATH**
- Download the repository and open a terminal inside it
- With **pip** just run ```pip install .``` to install the package
- You can see an example of how to use it in the [example](experiments/example.py).
- To run the main experiments you need another packages installed using :
  - ```pip install .[experiments]```

- Open a terminal in the experiments folder and run the file you want.

To compile on mac, add the following sections to your ~/.cargo/config (if you don't have this file feel free to create):
```
[target.x86_64-apple-darwin]
rustflags = [
"-C", "link-arg=-undefined",
"-C", "link-arg=dynamic_lookup",
]

[target.aarch64-apple-darwin]
rustflags = [
"-C", "link-arg=-undefined",
"-C", "link-arg=dynamic_lookup",
]
```
