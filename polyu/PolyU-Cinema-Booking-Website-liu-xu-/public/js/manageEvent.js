$(document).ready(function () {
  let seats = []; // Global variable to store seat data
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
    const rowsInput = document.getElementById('rows');
    const colsInput = document.getElementById('cols');
    const decreaseRows = document.getElementById('decrease-rows');
    const increaseRows = document.getElementById('increase-rows');
    const decreaseCols = document.getElementById('decrease-cols');
    const increaseCols = document.getElementById('increase-cols');
    const generateButton = document.getElementById('generate');
    const saveJsonButton = document.getElementById('save-json');
    const seatMap = document.getElementById('seat-map');
    const tooltip = document.getElementById('tooltip');

    // Helper function to update numeric input
    function updateInput(input, delta) {
      const currentValue = parseInt(input.value, 10) || 0;
      const newValue = Math.max(1, currentValue + delta);
      input.value = newValue;
    }

    // Increase/decrease row and column event listeners
    decreaseRows.addEventListener('click', () => updateInput(rowsInput, -1));
    increaseRows.addEventListener('click', () => updateInput(rowsInput, 1));
    decreaseCols.addEventListener('click', () => updateInput(colsInput, -1));
    increaseCols.addEventListener('click', () => updateInput(colsInput, 1));

    // Generate seat map
    generateButton.addEventListener('click', () => {
      const rows = parseInt(rowsInput.value, 10);
      const cols = parseInt(colsInput.value, 10);
      const seatSize = 40; // Size of each seat (rectangle)
      const seatSpacing = 10; // Spacing between seats
      const offsetX = 100; // X offset to leave space for row labels
      const offsetY = 60; // Y offset to leave space for screen label
      const totalSeats = rows * cols;

      // Generate random occupied seats (10% of total seats)
      const occupiedSeats = new Set();
      while (occupiedSeats.size < Math.ceil(totalSeats / 10)) {
        occupiedSeats.add(Math.floor(Math.random() * totalSeats));
      }

      // Clear previous SVG content
      seatMap.innerHTML = '';
      seatMap.setAttribute('height', offsetY + rows * (seatSize + seatSpacing) + 50);

      // Add screen label
      const screenLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      screenLabel.setAttribute('x', offsetX + (cols * (seatSize + seatSpacing)) / 2);
      screenLabel.setAttribute('y', 30);
      screenLabel.setAttribute('text-anchor', 'middle');
      screenLabel.setAttribute('font-size', '16');
      screenLabel.textContent = 'Screen';
      seatMap.appendChild(screenLabel);

      // Reset seats array
      seats = []; // Clear previous seat data

      // Generate seat data
      let seatNumber = 1;
      for (let row = 0; row < rows; row++) {
        const rowLabel = String.fromCharCode(65 + row); // Row labels: A, B, C, etc.

        // Row label at the start
        const startLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        startLabel.setAttribute('x', 10);
        startLabel.setAttribute('y', offsetY + row * (seatSize + seatSpacing) + seatSize / 2);
        startLabel.setAttribute('dy', '.35em');
        startLabel.textContent = rowLabel;
        seatMap.appendChild(startLabel);

        for (let col = 0; col < cols; col++) {
          const x = offsetX + col * (seatSize + seatSpacing);
          const y = offsetY + row * (seatSize + seatSpacing);
          const isOccupied = occupiedSeats.has(seatNumber - 1);
          const user = isOccupied ? `User${Math.floor(Math.random() * 1000)}` : null;

          const seat = { row: rowLabel, col: col + 1, seatNumber, user, occupy: isOccupied };
          seats.push(seat); // Add seat data to array

          // Add rectangle to SVG
          const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          rect.setAttribute('x', x);
          rect.setAttribute('y', y);
          rect.setAttribute('width', seatSize);
          rect.setAttribute('height', seatSize);
          rect.setAttribute('rx', 8); // Rounded corners
          rect.setAttribute('ry', 8);
          rect.setAttribute('fill', '#2ecc71'); // Color based on occupancy
          rect.setAttribute('stroke', '#2c3e50');
          rect.setAttribute('stroke-width', '2');
          rect.dataset.info = JSON.stringify(seat); // Attach seat info

          // Show tooltip on hover
          rect.addEventListener('mouseenter', (e) => {
            const info = JSON.parse(e.target.dataset.info); // 提取座位数据
          
            // 设置 tooltip 的内容
            tooltip.textContent = `Row: ${info.row}, Col: ${info.col}, Occupied: ${info.occupy}, User: ${info.user || 'None'}`;
          
            // 设置 tooltip 的位置
            tooltip.style.left = `${e.pageX + 10}px`; // 鼠标右侧 10px
            tooltip.style.top = `${e.pageY + 10}px`;  // 鼠标下方 10px
          
            // 显示 tooltip
            tooltip.style.visibility = 'visible'; 
          });
          
          rect.addEventListener('mouseleave', () => {
            tooltip.style.visibility = 'hidden'; // 鼠标移出时隐藏 tooltip
          });

          seatMap.appendChild(rect);

          // Add seat number
          const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
          text.setAttribute('x', x + seatSize / 2);
          text.setAttribute('y', y + seatSize / 2);
          text.setAttribute('dy', '.35em');
          text.setAttribute('text-anchor', 'middle');
          text.setAttribute('fill', '#fff');
          text.textContent = seatNumber++;
          seatMap.appendChild(text);
        }

        // Row label at the end
        const endLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        endLabel.setAttribute('x', offsetX + cols * (seatSize + seatSpacing) + 10);
        endLabel.setAttribute('y', offsetY + row * (seatSize + seatSpacing) + seatSize / 2);
        endLabel.setAttribute('dy', '.35em');
        endLabel.textContent = rowLabel;
        seatMap.appendChild(endLabel);
      }
    });

    // Save JSON as a file
    saveJsonButton.addEventListener('click', () => {
      const json = JSON.stringify(seats, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'seat-map.json';
      a.click();
      URL.revokeObjectURL(url);
    });
    $('#update').on('click', function () {
        if ($('#name').val() == '' || $('#price').val() == '') {
          alert('name and price cannot be empty');
        } 
        else {
          var formData = new FormData();
          formData.append('name2', `${$('#name').val()}`);
          formData.append('price', `${$('#price').val()}`);
          formData.append('date', `${$('#date').val()}`);
          formData.append('venue', `${$('#venue').val()}`);
          formData.append('introduction', `${$('#introduction').val()}`);
          alert(`./media/moviedata/${$('#name').val()}.jpg`);
          formData.append('image', `./media/moviedata/${$('#name').val()}.jpg`);
          var fileInput = $('#formFile')[0]; // 获取文件输入元素
          if (fileInput.files.length > 0) {
            formData.append('profileImage', fileInput.files[0]); // 添加图片文件到 FormData
          }
          $.ajax({
            url: '/admin/create',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
              if (data.status === 'success') {
                alert(`Update successfully}!`);
                window.open('/admin_show_movie.html', '_self');
              } else {
                alert(data.message);
              }
            },
            error: function (xhr) {
              const response = xhr.responseJSON || JSON.parse(xhr.responseText);
              if (response && response.message) {
                alert(response.message);
              } else {
                alert('Unknown error');
              }
              console.error('Error:', response);
            },
          });
        }
      });
});
