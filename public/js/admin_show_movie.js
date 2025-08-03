$(document).ready(function () {
    let events = []; // 全局变量，用于存储事件列表数据
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
    // 发送 AJAX 请求获取数据
    $.ajax({
      url: '/admin/list',
      type: 'GET',
      success: function (data) {
        if (data.status === 'success') {
          events = data.movielists.movies; // 获取事件列表数据
          console.log(events);

          // 渲染初始电影卡片
          $(events).each(function () {
            let innerhtml = `
              <div class="col-12 col-md-12 col-lg-6 card mb-2">
                <div class="row">
                  <div class="col-md-4">
                    <img src="${$(this)[0].image}" class="w-100 h-100 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${$(this)[0].name}" />
                  </div>
                  <div class="col-md-8">
                    <div class="card-body" style="text-align: left;">
                      <h6 class="card-title fw-bold">${$(this)[0].name}</h6>
                      <h4 class="badge text-bg-primary">${$(this)[0].type}</h4>
                      <p class="card-text">Price: ${$(this)[0].price}up</p>
                      <p class="card-text">Remaining Seats: ${$(this)[0].remainseat}</p>
                      <p class="card-text">Showing Date: ${this.date}</p>
                      <p class="card-text">Venue: ${$(this)[0].venue}</p>
                      <p style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${$(this)[0].introduction}</p>
                      <a onclick="gotodetails('${this.name}','${this.type}','${this.price}','${this.date}','${this.venue}','${this.image}')" class="nav-link link-primary" id="${this.id}">Manage</a>
                    </div>
                  </div>
                </div>
              </div>`;
            $('#movie_container').append(innerhtml);
          });

          // 绑定搜索框事件
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
        } else {
          alert('Please login');
          window.open('/login.html', '_self');
        }
      },
      error: function () {
        alert('Please login');
        window.open('/login.html', '_self');
      }
    });

    // 跳转到详情页面
    window.gotodetails = function (name,type,price,date,venue,image) {
      var formData = {
        name:name,
        type: type,
        price: price,
        date: date,
        image: image,
        venue: venue,
        // Additional fields can be added here
    };
      sessionStorage.setItem('movieData', JSON.stringify(formData));
      window.location.href = '/changeEvent.html';
    };
  });