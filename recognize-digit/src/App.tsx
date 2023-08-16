import React, { HTMLProps, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import { BarController } from 'chart.js';
import { eval_grid } from './draw';

const Cell: React.FC<
  {
    darkness: number;
  } & HTMLProps<HTMLDivElement>
> = ({ darkness, ...props }) => {
  return (
    <div
      className='grid-cell'
      style={{
        backgroundColor: `rgb(${darkness}, ${darkness}, ${darkness})`,
        border: '0.5px solid black',
      }}
      {...props}
    ></div>
  );
};

const Grid: React.FC<{}> = ({}) => {
  const N = 28;
  const [grid, setGrid] = useState<number[][]>(
    new Array(N).fill(0).map(() => new Array(N).fill(255))
  );

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
                    darkness={grid[rowId][colId]}
                    onMouseEnter={() => {
                      const newGrid = [...grid];
                      newGrid[rowId][colId] = 0;
                      setGrid(newGrid);
                    }}
                  ></Cell>
                );
              })}
            </div>
          );
        })}
      </div>
      <div style={{ display: ' flex', justifyContent: 'row' }}>
        {eval_grid(grid.flat()).map((res: number, ind: number) => {
          return <div style={{ fontSize: `${res * 20 + 4}px` }}>{ind}</div>;
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
