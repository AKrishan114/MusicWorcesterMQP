library(dplyr)
library(lubridate)

# Load data
data <- read.csv('/Users/akris/OneDrive/Desktop/Music Worcester Work/anon_DataMerge20250210 - anon_DataMerge.csv', 
                 stringsAsFactors = FALSE)

# Convert to Date format
data$CreatedDate <- as.Date(data$CreatedDate, format = "%m/%d/%Y")
data$EventDate <- as.Date(data$EventDate, format = "%m/%d/%Y")

# Filter data
filtered_data <- data %>%
  filter(
    CreatedDate <= EventDate,
    year(EventDate) < 2026,
    year(CreatedDate) <= 2024,
    Subscriber == 'never',               
    !ChorusMember,
    !Student,
    EventType != 'Virtual'
  )

# Daily ticket counts
daily_ticket_counts <- filtered_data %>%
  group_by(CreatedDate, EventClass) %>%
  summarise(TicketCount = n(), .groups = 'drop')

# Fill missing dates
all_dates <- seq(min(filtered_data$CreatedDate, na.rm = TRUE),
                 max(filtered_data$CreatedDate, na.rm = TRUE), by = "day")

all_classes <- unique(filtered_data$EventClass)

daily_ticket_counts <- expand.grid(Date = all_dates, EventClass = all_classes) %>%
  left_join(daily_ticket_counts, by = c("Date" = "CreatedDate", "EventClass" = "EventClass")) %>%
  mutate(TicketCount = ifelse(is.na(TicketCount), 0, TicketCount))

# Monthly ticket counts by EventClass, excluding summer months (June, July, August)
monthly_ticket_counts <- daily_ticket_counts %>%
  mutate(Month = floor_date(Date, "month")) %>%
  filter(!(month(Month) %in% c(6, 7, 8))) %>%
  group_by(Month) %>% #EventClass here for EventClass
  summarise(TicketCount = sum(TicketCount), .groups = 'drop')

# Define periods
periods <- list(
  "Pre-Pandemic" = filter(monthly_ticket_counts, Month > as.Date("2013-09-01") & Month < as.Date("2020-03-01")),
  "During Pandemic" = filter(monthly_ticket_counts, Month >= as.Date("2020-03-01") & Month < as.Date("2021-03-01")),
  "Post-Pandemic" = filter(monthly_ticket_counts, Month >= as.Date("2021-03-01"))
)

# Function to get stats by EventClass
get_stats_by_class <- function(data) {
  data %>%
    #group_by(EventClass) %>% #group_by(EventClass)
    summarise(
      Mean = mean(TicketCount, na.rm = TRUE),
      Median = median(TicketCount, na.rm = TRUE),
      Mode = as.numeric(names(sort(table(TicketCount), decreasing = TRUE)[1])),
      Q1 = quantile(TicketCount, 0.25, na.rm = TRUE),
      Q3 = quantile(TicketCount, 0.75, na.rm = TRUE),
      Min = min(TicketCount, na.rm = TRUE),
      Max = max(TicketCount, na.rm = TRUE),
      .groups = 'drop'
    )
}

# Calculate statistics for each period by EventClass
statistics_by_class <- lapply(periods, get_stats_by_class)

# Print results
for (period in names(statistics_by_class)) {
  cat("\n", period, ":\n")
  print(statistics_by_class[[period]])
}

