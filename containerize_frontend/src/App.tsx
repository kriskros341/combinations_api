import React, { useRef, useState } from 'react'

type InputChange = React.ChangeEvent<HTMLInputElement>

const ContainerSizeInput = (
  {currentValue, changeHandler, remove}: {currentValue: number, changeHandler: (value: number) => void, remove: () => void}
  ) => {
    const ref = useRef<HTMLInputElement>(null)
    if(ref.current) {
      ref.current.value = currentValue.toString()
    }
  return (
    <div>
      <input defaultValue={currentValue} onChange={(e: InputChange) => {
        let value = parseInt(e.target.value)
        if(!isNaN(value)) {
          changeHandler(parseInt(e.target.value))
        }
      }} type="number"></input>
      <button onClick={remove}>x</button>
    </div>
  )
}


function App() {
  const [count, setCount] = useState<number[][]>([])
  const [found, setFound] = useState<number>(0)
  const sumsInputValueRef = useRef(0)
  const [_, rerenderer] = useState(false)
  const rerender = () => rerenderer(v => !v)

  const containerSizesRef = useRef<number[]>([]) 
  //const calculateButtonHandler = () => {
  //  let argsList = "args="
  //  argsList += containerSizesRef.current.join("&args=")
  //  fetch(`http://localhost:2223/?${argsList}&sums=${sumsInputValueRef.current}`)
  //    .then(response => response.json())
  //    .then(data => setCount(JSON.parse(data.result)))
  //    .then(() => rerender())
  
  
  async function handleStreamResponse(r: Response) {
    let c = await r.text()
    c = "["+c+"[]]" //  Because JSON 
    const values = JSON.parse(c).slice(0, -1)
    setCount(v => [...v, ...values])
    setFound(v => v + values.length)
  }

  async function calculateButtonHandler() {
    setCount([])
    setFound(0)
    let argsList = "args="
    argsList += containerSizesRef.current.join("&args=")
    const res = await fetch(`http://localhost:2223/stream/?${argsList}&sums=${sumsInputValueRef.current}`)
    const reader = res.body?.getReader()
    if (!reader) {
      return
    }
    while(true) {
      const {done, value} = await reader.read()
      if (done) {
        break;
      }
      const response = new Response(value)
      handleStreamResponse(response)
    }
  }

  const inputHandler1 = (e: InputChange) => sumsInputValueRef.current = parseInt(e.target.value)
  const addContainerButtonHandler = () => {
    containerSizesRef.current.push(0)
    rerender()
  }
  const removeContainer = (idx: number) => {
    containerSizesRef.current = [...containerSizesRef.current.slice(0, idx), ...containerSizesRef.current.slice(idx+1)]
    rerender()
  }
  const changeContainerHandler = (idx: number, value: number) => {
    containerSizesRef.current = [...containerSizesRef.current.slice(0, idx), value, ...containerSizesRef.current.slice(idx+1)]
  }
  return (
    <div>
      <input defaultValue={0} onChange={inputHandler1} type="number"></input><br/>      
      -----------------------------------------------------------
      {containerSizesRef.current.length == 0 ? 
        <div>no container sizes</div> 
        : 
        containerSizesRef.current.map((value, index) => 
          <ContainerSizeInput
            key={`container ${index} with ${value}`}
            currentValue={value} 
            changeHandler={(newValue) => 
              changeContainerHandler(index, newValue)
            } 
            remove={() => removeContainer(index)}
          />
        )
      }
      <button onClick={addContainerButtonHandler}>add container size</button>
      {containerSizesRef.current.length > 0 && <button onClick={calculateButtonHandler}>Calculate</button>}<br/>
      -----------------------------------------------------------
      {count.length != 0 && <div>rezultaty dla sumy {sumsInputValueRef.current}: ilość {found}</div>}
      <table>
        <thead>
          <tr>
            {containerSizesRef.current.map((value, idx) => <th key={`header_${idx}:${value}`}>{value}</th>)}
          </tr>
        </thead>
        <tbody>
          {count.slice(0, 100).map((rowData, rowIdx) => 
            <tr key={`row_${rowIdx}:${rowData}`}>
                {rowData.map((counter, idx) => <td key={`counter_${idx}:${counter}`}>{counter}</td>)}
              <td>sums up to: 
                {rowData.map((nextVal, idx) => nextVal * containerSizesRef.current[idx]).reduce((val, total) => total + val)}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

export default App
