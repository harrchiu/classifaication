import React, { HTMLProps, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { BarController } from "chart.js";
import { eval_grid } from "./draw";

const CLEAR_KEY = "c";

const Cell: React.FC<
  {
    value: number;
  } & HTMLProps<HTMLDivElement>
> = ({ value, ...props }) => {
  const darkness = 255 - value;
  return (
    <div
      className="grid-cell"
      style={{
        backgroundColor: `rgb(${darkness}, ${darkness}, ${darkness})`,
        // border: '0.1px solid black',
      }}
      {...props}
    ></div>
  );
};

// test again
const Grid: React.FC<{}> = ({}) => {
  const sz = 28;
  const penExtra = 1;
  const scale = 1;
  const N = scale * 28;
  const [grid, setGrid] = useState<number[][]>(
    new Array(N).fill(0).map(() => new Array(N).fill(0))
  );
  const [isDrawOn, setIsDrawOn] = useState(false);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === " ") {
      setIsDrawOn(true);
    } else if (e.key === CLEAR_KEY) {
      setGrid(new Array(N).fill(0).map(() => new Array(N).fill(0)));
    }
  };

  const handleKeyUp = (e: KeyboardEvent) => {
    if (e.key === " ") {
      setIsDrawOn(false);
    }
  };

  const handleMouseDown = (e: MouseEvent) => {
    setIsDrawOn(true);
  };
  const handleMouseUp = (e: MouseEvent) => {
    setIsDrawOn(false);
  };

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
      document.addEventListener("mousedown", handleMouseDown);
      document.addEventListener("mouseup", handleMouseUp);
    };
  });

  const predict = () => {
    const shortened = new Array(sz).fill(0).map(() => new Array(sz).fill(0));
    for (let r = 0; r < sz; r += 1) {
      for (let c = 0; c < sz; c += 1) {
        for (let a = 0; a < scale; a += 1) {
          for (let b = 0; b < scale; b += 1) {
            shortened[r][c] += grid[r * scale + a][c * scale + b];
          }
        }
        shortened[r][c] /= scale * scale;
      }
    }
    return eval_grid([shortened.flat()]);
  };

  return (
    <>
      <div className="drawing-grid">
        <div
          style={{
            marginBottom: "5px",
            display: "flex",
            alignItems: "end",
            flexDirection: "row",
            justifyContent: "space-between",
            gap: "5px",
          }}
        >
          <div>
            <a
              href="https://github.com/harrchiu/classifaication"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
            <div>{`<${CLEAR_KEY}>`} to clear</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div>Click or hold {`<space>`} to draw</div>
          </div>
        </div>
        {grid.map((row, rowId) => {
          return (
            <div className="drawing-grid__row" key={`row-${rowId}`}>
              {row.map((cell, colId) => {
                return (
                  <Cell
                    key={`cell-${rowId}-${colId}`}
                    value={grid[rowId][colId]}
                    onMouseEnter={() => {
                      if (!isDrawOn) {
                        return;
                      }
                      const newGrid = [...grid];
                      newGrid[rowId][colId] = 255;

                      for (let r = -penExtra; r < penExtra; r += 1) {
                        for (let c = -penExtra; c < penExtra; c += 1) {
                          if (newGrid[rowId + r][colId + c] === 0) {
                            newGrid[rowId + r][colId + c] =
                              200 + Math.random() * 55;
                          }
                        }
                      }
                      setGrid(newGrid);
                    }}
                  ></Cell>
                );
              })}
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", height: "40px", justifyContent: "row" }}>
        {predict().map((res: number, ind: number) => {
          return (
            <div key={`dig-${ind}`} style={{ fontSize: `${res * 20 + 4}px` }}>
              {ind}
            </div>
          );
        })}
      </div>
    </>
  );
};

const App: React.FC<{}> = () => {
  return (
    <div className="app-page">
      <Grid />
      <div style={{ textAlign: "left", fontSize: "11px" }}>
        For best results, write large, write centred, and just write smth from
        the{" "}
        <a
          href="https://www.google.com/search?sca_esv=558970703&q=mnist+dataset&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjK9Innt--AAxVRl4kEHUOWDBUQ0pQJegQIDBAB&biw=1582&bih=871&dpr=2#imgrc=u8BarxpjCS7zIM"
          target="_blank"
          rel="noreferrer"
        >
          dataset
        </a>
      </div>
      {/* <BarController></BarController> */}
    </div>
  );
};

export default App;
