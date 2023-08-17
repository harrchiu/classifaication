import React, { HTMLProps, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import { BarController } from 'chart.js';
import { eval_grid } from './draw';

const Cell: React.FC<
  {
    value: number;
  } & HTMLProps<HTMLDivElement>
> = ({ value, ...props }) => {
  const darkness = 255 - value;
  return (
    <div
      className='grid-cell'
      style={{
        backgroundColor: `rgb(${darkness}, ${darkness}, ${darkness})`,
        // border: '0.1px solid black',
      }}
      {...props}
    ></div>
  );
};

const Grid: React.FC<{}> = ({}) => {
  const sz = 28;
  const scale = 1;
  const N = scale * 28;
  const [grid, setGrid] = useState<number[][]>(
    new Array(N).fill(0).map(() => new Array(N).fill(0))
  );

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
      <div className='drawing-grid'>
        {grid.map((row, rowId) => {
          return (
            <div className='drawing-grid__row' key={`row-${rowId}`}>
              {row.map((cell, colId) => {
                return (
                  <Cell
                    key={`cell-${rowId}-${colId}`}
                    value={grid[rowId][colId]}
                    onMouseEnter={() => {
                      const newGrid = [...grid];
                      newGrid[rowId][colId] = 255;
                      setGrid(newGrid);
                      console.log(newGrid.flat());
                    }}
                  ></Cell>
                );
              })}
            </div>
          );
        })}
      </div>
      <div style={{ display: ' flex', height: '40px', justifyContent: 'row' }}>
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
    <div className='app-page'>
      <Grid />
      {/* <BarController></BarController> */}
    </div>
  );
};

export default App;
