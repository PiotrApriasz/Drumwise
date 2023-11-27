using System.Reflection;
using Ardalis.GuardClauses;
using Drumwise.Application.Common.Behaviours;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Data.Interceptors;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using FluentValidation;
using MediatR;
using MediatR.Pipeline;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;

namespace Drumwise.API;

public static class ServicesConfigurator
{
    public static IServiceCollection ConfigureIdentity(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("IdentityConnection");
        Guard.Against.Null(connectionString, $"Connection string for 'IdentityConnection' not found");
        
        services.AddDbContext<AppIdentityDbContext>((sp, options) =>
        {
            // Will use mongoDb database with new Entity Framework MongoDb provider
            //options.UseSqlite(connectionString); 
        });

        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<AppIdentityDbContext>());
        services.AddScoped<AppIdentityDbContextInitializer>();
        
        services.AddIdentity<ApplicationUser, Roles>()
            .AddRoles<IdentityRole>()
            .AddEntityFrameworkStores<AppIdentityDbContext>();
        
        services.AddTransient<IIdentityService, IdentityService>();
        
        services.AddAuthorizationBuilder()
                    .AddPolicy(Policies.CanAddHomework, policy => policy.RequireRole(Roles.Teacher));

        return services;
    }
    
    public static IServiceCollection AddInfrastructuresServices(this IServiceCollection services, IConfiguration configuration)
    {
        var connectionString = configuration.GetConnectionString("AppConnection");
        Guard.Against.Null(connectionString, $"Connection string for 'AppConnection' not found");

        services.AddScoped<ISaveChangesInterceptor, AuditableEntityInterceptor>();
        services.AddScoped<ISaveChangesInterceptor, DispatchDomainEventsInterceptor>();

        services.AddDbContext<ApplicationDbContext>((sp, options) =>
        {
            options.AddInterceptors(sp.GetService<ISaveChangesInterceptor>()!);
            options.UseSqlite(connectionString);
        });

        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<ApplicationDbContext>());
        services.AddScoped<ApplicationDbContextInitializer>();

        return services;
    }

    public static IServiceCollection AddApplicationServices(this IServiceCollection services)
    {
        services.AddAutoMapper(Assembly.GetExecutingAssembly());

        services.AddValidatorsFromAssembly(Assembly.GetExecutingAssembly());

        services.AddMediatR(cfg =>
        {
            cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly());
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(UnhandledExceptionBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(AuthorizationBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(ValidationBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(PerformanceBahaviour<,>));
            cfg.AddBehavior(typeof(IRequestPreProcessor<>), typeof(LoggingBehaviour<>));
        });

        return services;
    }
}