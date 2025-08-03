$(document).ready(function () {
  // <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
  let events = []; // 全局变量，用于存储事件列表数据
    $.get('./media/jsondata/movielist.json', function (data) {
      events=data;
        $(data).each(function () {
            let innerhtml = `
            <div class="col-12 col-md-12 col-lg-6 card mb-2 ">
                <div class="row ">
                    <div class="col-md-4 ">
                        <img src="${$(this)[0].image}" class="w-100 h-100 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${$(this)[0].name}" style="height:100%">
                    </div>
                    <div class="col-md-8">
                        <div class="card-body">
                            <h6 class="card-title fw-bold">${$(this)[0].name}</h6>
                            <h4 class="badge text-bg-primary">${$(this)[0].type}</h4>
                            <p class="card-text">price:${$(this)[0].price}up</p>`
           if($(this)[0].remainseat==0)                 
          innerhtml = innerhtml+`<p class="card-text" style="color:red">Remaining Seats: NOT Available</p>
          <p style="overflow: hidden;text-overflow: ellipsis;white-space: nowrap;">${$(this)[0].introduction}</p>
                            <a onclick="nota()" class="nav-link link-primary" id="${this.id}">Details</a>
                        </div>
                    </div>
                </div>
            </div>`;
          else
          innerhtml = innerhtml+ `<p class="card-text">Remaining Seats: ${$(this)[0].remainseat}</p><p class="card-text">Showing date: ${$(this)[0].data}</p><p style="overflow: hidden;text-overflow: ellipsis;white-space: nowrap;">${$(this)[0].introduction}</p>
                            <a onclick="gotodetails('${this.id}')" class="nav-link link-primary" id="${this.id}">Details</a>
                        </div>
                    </div>
                </div>
            </div>`;
                            
            $('#movie_container').append(innerhtml);
        });
        const names = [
            "Chinese Ethnic Song and Dance Gala",
            "Chang An",
            "THE LAST DANCE"
          ];
          const dic = {
            "Chinese Ethnic Song and Dance Gala":"polyu01",
            "Chang An":"polyu02",
            "THE LAST DANCE":"polyu03"
        };    
          const input = document.getElementById("search-input");
          const suggestionsList = document.getElementById("suggestions");
          const redirectButton = document.getElementById("redirect-button");
          let selectedName = ""; // 存储选中的名字
      
          // 监听输入框的输入事件
          input.addEventListener("input", () => {
            const query = input.value.toLowerCase().trim();
      
            // 如果输入为空，清空建议列表
            if (!query) {
              suggestionsList.innerHTML = "";
              redirectButton.disabled = true; // 禁用按钮
              selectedName = ""; // 清空选中状态
              return;
            }
      
            // 从列表中匹配包含输入内容的名字
            const matches = names.filter(name =>
              name.toLowerCase().includes(query)
            );
      
            // 渲染建议列表
            suggestionsList.innerHTML = matches
              .map(name => `<li>${name}</li>`)
              .join("");
      
            // 为每个建议项添加点击事件
            Array.from(suggestionsList.children).forEach(item => {
              item.addEventListener("click", () => {
                input.value = item.textContent; // 将选中内容填充到输入框
                selectedName = item.textContent; // 保存选中的名字
                suggestionsList.innerHTML = ""; // 清空建议列表
                redirectButton.disabled = false; // 启用按钮
              });
            });
          });
          $('#name-search, #start-time, #end-time, #venue-search,#Description').on('input change', function () {
            const nameQuery = $('#name-search').val().toLowerCase().trim();
            const startTime = $('#start-time').val();
            const endTime = $('#end-time').val();
            const venueQuery = $('#venue-search').val().toLowerCase().trim();
            var Description = $('#Description').val().toLowerCase().trim();
            // 筛选符合条件的事件
            const filteredEvents = events.filter(event => {
              const matchesName = !nameQuery || event.name.toLowerCase().includes(nameQuery);
              const eventTime = new Date(event.date).getTime();
              const matchesStartTime = !startTime || eventTime >= new Date(startTime).getTime();
              const matchesEndTime = !endTime || eventTime <= new Date(endTime).getTime();
              const matchesVenue = !venueQuery || event.venue.toLowerCase().includes(venueQuery);
              Description = !Description || event.venue.toLowerCase().includes(Description);
              return matchesName && matchesStartTime && matchesEndTime && matchesVenue&&Description;
            });

            // 渲染筛选结果
            $('#event-list').html(
              filteredEvents.length
                ? filteredEvents
                    .map(
                      event => `
                      <li>
                        <a href="${event.link}" target="_blank">${event.name}</a>
                      </li>
                    `
                    )
                    .join('')
                : '<p>No events found.</p>'
            );
          });

          // 清空按钮功能
          $('#clear-button').on('click', function () {
            $('#name-search').val('');
            $('#start-time').val('');
            $('#end-time').val('');
            $('#venue-search').val('');
            $('#event-list').html(''); // 清空结果
          });
          // 跳转按钮事件
          redirectButton.addEventListener("click", () => {
            if (selectedName) {
                gotodetails(dic[$("#search-input").val()]);
            }
          });
    }).fail(function (error) {
        let innerhtml = `
         <div class="col mb-3">
            <div class="alert alert-danger" role="alert">
                Failed to fetch drink menu. Please try again later.
            </div>
        </div>`;
        $('#menu_container').append(innerhtml);
    });
    
});
function gotodetails(id) {
    var formData = new FormData();
    formData.append('id', id);
    $.ajax({
        url: '/event/movie',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            if (data.status === 'success') {
            window.location.href = '/event_detail.html';
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
            window.location.href = '/login.html';
        },
    });
  };
  function nota(){
    alert("There is no seat remains,Select another one");
  }
